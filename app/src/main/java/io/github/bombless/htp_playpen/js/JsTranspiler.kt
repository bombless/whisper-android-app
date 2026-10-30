package io.github.bombless.htp_playpen.js

/**
 * Lightweight JavaScript preprocessor that transpiles ES6 class declarations
 * into prototype-based function definitions supported by Mozilla Rhino.
 */
object JsTranspiler {

    fun transpile(source: String): String {
        if (!source.contains("class ")) return source

        val output = StringBuilder()
        var index = 0
        val len = source.length

        while (index < len) {
            val classKeywordIdx = findClassKeyword(source, index)
            if (classKeywordIdx == -1) {
                output.append(source.substring(index))
                break
            }

            output.append(source.substring(index, classKeywordIdx))
            val result = parseClass(source, classKeywordIdx)
            if (result != null) {
                output.append(result.transpiledCode)
                index = result.endIndex
            } else {
                // If parsing failed, just output the "class" and move forward
                output.append("class")
                index = classKeywordIdx + 5
            }
        }

        return output.toString()
    }

    private data class ClassParseResult(val transpiledCode: String, val endIndex: Int)

    private fun findClassKeyword(source: String, start: Int): Int {
        var i = start
        var inSingleQuote = false
        var inDoubleQuote = false
        var inBacktick = false
        var inLineComment = false
        var inBlockComment = false

        while (i < source.length - 5) {
            val c = source[i]
            val next = source[i + 1]

            if (inLineComment) {
                if (c == '\n') inLineComment = false
                i++
                continue
            }
            if (inBlockComment) {
                if (c == '*' && next == '/') {
                    inBlockComment = false
                    i += 2
                    continue
                }
                i++
                continue
            }
            if (inSingleQuote) {
                if (c == '\\') i += 2 else { if (c == '\'') inSingleQuote = false; i++ }
                continue
            }
            if (inDoubleQuote) {
                if (c == '\\') i += 2 else { if (c == '"') inDoubleQuote = false; i++ }
                continue
            }
            if (inBacktick) {
                if (c == '\\') i += 2 else { if (c == '`') inBacktick = false; i++ }
                continue
            }

            if (c == '/' && next == '/') { inLineComment = true; i += 2; continue }
            if (c == '/' && next == '*') { inBlockComment = true; i += 2; continue }
            if (c == '\'') { inSingleQuote = true; i++; continue }
            if (c == '"') { inDoubleQuote = true; i++; continue }
            if (c == '`') { inBacktick = true; i++; continue }

            // Check for word boundary before "class "
            val prevChar = if (i > 0) source[i - 1] else ' '
            if (!prevChar.isJavaIdentifierPart() && source.startsWith("class ", i)) {
                return i
            }
            i++
        }
        return -1
    }

    private fun parseClass(source: String, start: Int): ClassParseResult? {
        var i = start + 5 // skip "class"
        while (i < source.length && source[i].isWhitespace()) i++

        // Extract class name
        val nameStart = i
        while (i < source.length && (source[i].isJavaIdentifierPart() || source[i] == '$')) i++
        val className = source.substring(nameStart, i).trim()
        if (className.isEmpty()) return null

        while (i < source.length && source[i].isWhitespace()) i++

        // Optional extends
        var parentClass: String? = null
        if (source.startsWith("extends", i) && (i + 7 >= source.length || !source[i + 7].isJavaIdentifierPart())) {
            i += 7
            while (i < source.length && source[i].isWhitespace()) i++
            val parentStart = i
            while (i < source.length && (source[i].isJavaIdentifierPart() || source[i] == '$')) i++
            parentClass = source.substring(parentStart, i).trim()
            while (i < source.length && source[i].isWhitespace()) i++
        }

        if (i >= source.length || source[i] != '{') return null
        val bodyStart = i + 1

        // Find matching closing brace
        val bodyEnd = findMatchingBrace(source, i)
        if (bodyEnd == -1) return null

        val body = source.substring(bodyStart, bodyEnd)
        val transpiled = transpileClassBody(className, parentClass, body)
        return ClassParseResult(transpiled, bodyEnd + 1)
    }

    private fun findMatchingBrace(source: String, openBraceIdx: Int): Int {
        var depth = 0
        var i = openBraceIdx
        var inSingleQuote = false
        var inDoubleQuote = false
        var inBacktick = false
        var inLineComment = false
        var inBlockComment = false

        while (i < source.length) {
            val c = source[i]
            val next = if (i + 1 < source.length) source[i + 1] else ' '

            if (inLineComment) {
                if (c == '\n') inLineComment = false
                i++
                continue
            }
            if (inBlockComment) {
                if (c == '*' && next == '/') {
                    inBlockComment = false
                    i += 2
                    continue
                }
                i++
                continue
            }
            if (inSingleQuote) {
                if (c == '\\') i += 2 else { if (c == '\'') inSingleQuote = false; i++ }
                continue
            }
            if (inDoubleQuote) {
                if (c == '\\') i += 2 else { if (c == '"') inDoubleQuote = false; i++ }
                continue
            }
            if (inBacktick) {
                if (c == '\\') i += 2 else { if (c == '`') inBacktick = false; i++ }
                continue
            }

            if (c == '/' && next == '/') { inLineComment = true; i += 2; continue }
            if (c == '/' && next == '*') { inBlockComment = true; i += 2; continue }
            if (c == '\'') { inSingleQuote = true; i++; continue }
            if (c == '"') { inDoubleQuote = true; i++; continue }
            if (c == '`') { inBacktick = true; i++; continue }

            if (c == '{') {
                depth++
            } else if (c == '}') {
                depth--
                if (depth == 0) return i
            }
            i++
        }
        return -1
    }

    private fun transpileClassBody(className: String, parentClass: String?, body: String): String {
        var constructorParams = ""
        var constructorBody: String? = null
        val instanceMethods = mutableListOf<Triple<String, String, String>>() // name, params, body
        val staticMethods = mutableListOf<Triple<String, String, String>>() // name, params, body
        val fieldInits = mutableListOf<String>()

        var i = 0
        val len = body.length

        while (i < len) {
            while (i < len && body[i].isWhitespace()) i++
            if (i >= len) break

            // Skip comments
            if (body[i] == '/' && i + 1 < len && body[i + 1] == '/') {
                while (i < len && body[i] != '\n') i++
                continue
            }
            if (body[i] == '/' && i + 1 < len && body[i + 1] == '*') {
                i += 2
                while (i < len - 1 && !(body[i] == '*' && body[i + 1] == '/')) i++
                i += 2
                continue
            }

            var isStatic = false
            if (body.startsWith("static", i) && i + 6 < len && body[i + 6].isWhitespace()) {
                isStatic = true
                i += 6
                while (i < len && body[i].isWhitespace()) i++
            }

            // Identifier (name)
            val nameStart = i
            while (i < len && (body[i].isJavaIdentifierPart() || body[i] == '$')) i++
            val memberName = body.substring(nameStart, i).trim()
            if (memberName.isEmpty()) {
                i++
                continue
            }

            while (i < len && body[i].isWhitespace()) i++

            // Check if this is a method: followed by '('
            if (i < len && body[i] == '(') {
                // Parse parameter list
                val paramStart = i + 1
                val paramEnd = findMatchingParen(body, i)
                if (paramEnd == -1) break
                val params = body.substring(paramStart, paramEnd).trim()
                i = paramEnd + 1
                while (i < len && body[i].isWhitespace()) i++

                if (i < len && body[i] == '{') {
                    val methodBodyEnd = findMatchingBrace(body, i)
                    if (methodBodyEnd == -1) break
                    val mBody = body.substring(i + 1, methodBodyEnd)
                    i = methodBodyEnd + 1

                    if (memberName == "constructor" && !isStatic) {
                        constructorParams = params
                        constructorBody = mBody
                    } else if (isStatic) {
                        staticMethods.add(Triple(memberName, params, mBody))
                    } else {
                        instanceMethods.add(Triple(memberName, params, mBody))
                    }
                }
            } else if (i < len && (body[i] == '=' || body[i] == ';')) {
                // Class field declaration, e.g. root = null;
                val fieldEnd = body.indexOf(';', i).let { if (it == -1) len else it }
                val initExpr = body.substring(nameStart, fieldEnd).trim()
                if (isStatic) {
                    fieldInits.add("$className.$initExpr;")
                } else {
                    fieldInits.add("this.$initExpr;")
                }
                i = fieldEnd + 1
            } else {
                i++
            }
        }

        val sb = StringBuilder()
        sb.append("\n// Transpiled class $className\n")

        // Constructor function
        sb.append("function ").append(className).append("(").append(constructorParams).append(") {\n")
        if (parentClass != null && (constructorBody == null || !constructorBody.contains("super("))) {
            sb.append("    $parentClass.apply(this, arguments);\n")
        }
        for (field in fieldInits.filter { it.startsWith("this.") }) {
            sb.append("    ").append(field).append("\n")
        }
        if (constructorBody != null) {
            val adjustedBody = if (parentClass != null) {
                constructorBody.replace(Regex("""super\s*\((.*?)\);?"""), "$parentClass.call(this, $1);")
            } else {
                constructorBody
            }
            sb.append(adjustedBody)
        }
        sb.append("\n}\n")

        // Prototype inheritance if extends
        if (parentClass != null) {
            sb.append("if (typeof ").append(parentClass).append(" !== 'undefined') {\n")
            sb.append("    ").append(className).append(".prototype = Object.create(").append(parentClass).append(".prototype);\n")
            sb.append("    ").append(className).append(".prototype.constructor = ").append(className).append(";\n")
            sb.append("    Object.setPrototypeOf(").append(className).append(", ").append(parentClass).append(");\n")
            sb.append("}\n")
        }

        // Instance methods
        for ((mName, mParams, mBody) in instanceMethods) {
            sb.append(className).append(".prototype.").append(mName).append(" = function(").append(mParams).append(") {\n")
            sb.append(mBody)
            sb.append("\n};\n")
        }

        // Static methods and static fields
        for ((mName, mParams, mBody) in staticMethods) {
            sb.append(className).append(".").append(mName).append(" = function(").append(mParams).append(") {\n")
            sb.append(mBody)
            sb.append("\n};\n")
        }
        for (field in fieldInits.filter { it.startsWith("$className.") }) {
            sb.append(field).append("\n")
        }

        return sb.toString()
    }

    private fun findMatchingParen(source: String, openParenIdx: Int): Int {
        var depth = 0
        var i = openParenIdx
        while (i < source.length) {
            val c = source[i]
            if (c == '(') depth++
            else if (c == ')') {
                depth--
                if (depth == 0) return i
            }
            i++
        }
        return -1
    }
}
