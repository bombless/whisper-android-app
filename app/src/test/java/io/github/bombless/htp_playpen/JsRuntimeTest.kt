package io.github.bombless.htp_playpen

import io.github.bombless.htp_playpen.js.JsRuntime
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class JsRuntimeTest {

    @Test
    fun testBasicArithmeticAndReturn() {
        val result = JsRuntime.execute("const a = 10; const b = 32; a + b;")
        assertTrue("Execution should succeed: ${result.error}", result.success)
        assertEquals("42", result.result)
    }

    @Test
    fun testConsoleLogCapture() {
        val script = """
            console.log("Hello", "World!");
            console.info("Status:", 200);
            console.log("Array:", [1, 2, 3]);
            "finished";
        """.trimIndent()

        val result = JsRuntime.execute(script)
        assertTrue(result.success)
        assertEquals("finished", result.result)
        val logs = result.stdout.lines()
        assertEquals(3, logs.size)
        assertEquals("Hello World!", logs[0])
        assertEquals("Status: 200", logs[1])
        assertTrue("Should format array as JSON", logs[2].contains("[1,2,3]"))
    }

    @Test
    fun testAssertionSuccessAndFailure() {
        val passScript = """
            assert(1 + 1 === 2, "Math should work");
            console.log("Passed math check");
        """.trimIndent()
        val passResult = JsRuntime.execute(passScript)
        assertTrue(passResult.success)
        assertTrue(passResult.stdout.contains("Passed math check"))

        val failScript = """
            assert(2 + 2 === 5, "Two plus two is four, not five");
        """.trimIndent()
        val failResult = JsRuntime.execute(failScript)
        assertFalse(failResult.success)
        assertTrue(failResult.error?.contains("Two plus two is four, not five") == true)
    }

    @Test
    fun testRedBlackTreeImplementationAndVerification() {
        val rbTreeScript = """
            const RED = true;
            const BLACK = false;

            class Node {
                constructor(key, color = RED) {
                    this.key = key;
                    this.color = color;
                    this.left = null;
                    this.right = null;
                    this.parent = null;
                }
            }

            class RedBlackTree {
                constructor() {
                    this.root = null;
                }

                rotateLeft(x) {
                    const y = x.right;
                    x.right = y.left;
                    if (y.left !== null) y.left.parent = x;
                    y.parent = x.parent;
                    if (x.parent === null) {
                        this.root = y;
                    } else if (x === x.parent.left) {
                        x.parent.left = y;
                    } else {
                        x.parent.right = y;
                    }
                    y.left = x;
                    x.parent = y;
                }

                rotateRight(y) {
                    const x = y.left;
                    y.left = x.right;
                    if (x.right !== null) x.right.parent = y;
                    x.parent = y.parent;
                    if (y.parent === null) {
                        this.root = x;
                    } else if (y === y.parent.right) {
                        y.parent.right = x;
                    } else {
                        y.parent.left = x;
                    }
                    x.right = y;
                    y.parent = x;
                }

                insert(key) {
                    let node = new Node(key);
                    let y = null;
                    let x = this.root;
                    while (x !== null) {
                        y = x;
                        if (node.key < x.key) x = x.left;
                        else x = x.right;
                    }
                    node.parent = y;
                    if (y === null) {
                        this.root = node;
                    } else if (node.key < y.key) {
                        y.left = node;
                    } else {
                        y.right = node;
                    }
                    this.fixInsert(node);
                }

                fixInsert(z) {
                    while (z.parent && z.parent.color === RED) {
                        if (z.parent === z.parent.parent.left) {
                            let y = z.parent.parent.right;
                            if (y && y.color === RED) {
                                z.parent.color = BLACK;
                                y.color = BLACK;
                                z.parent.parent.color = RED;
                                z = z.parent.parent;
                            } else {
                                if (z === z.parent.right) {
                                    z = z.parent;
                                    this.rotateLeft(z);
                                }
                                z.parent.color = BLACK;
                                z.parent.parent.color = RED;
                                this.rotateRight(z.parent.parent);
                            }
                        } else {
                            let y = z.parent.parent.left;
                            if (y && y.color === RED) {
                                z.parent.color = BLACK;
                                y.color = BLACK;
                                z.parent.parent.color = RED;
                                z = z.parent.parent;
                            } else {
                                if (z === z.parent.left) {
                                    z = z.parent;
                                    this.rotateRight(z);
                                }
                                z.parent.color = BLACK;
                                z.parent.parent.color = RED;
                                this.rotateLeft(z.parent.parent);
                            }
                        }
                    }
                    this.root.color = BLACK;
                }

                // Verify RB-Tree properties:
                // 1. Root is black
                // 2. Red nodes cannot have red children
                // 3. Black height is equal on all root-to-leaf paths
                verify() {
                    if (this.root === null) return true;
                    assert(this.root.color === BLACK, "Root must be black");

                    function checkNode(node) {
                        if (node === null) return 1; // null leaf is black
                        if (node.color === RED) {
                            assert(!node.left || node.left.color === BLACK, "Red node cannot have red left child");
                            assert(!node.right || node.right.color === BLACK, "Red node cannot have red right child");
                        }
                        const leftHeight = checkNode(node.left);
                        const rightHeight = checkNode(node.right);
                        assert(leftHeight === rightHeight, "Black height violation at node " + node.key);
                        return leftHeight + (node.color === BLACK ? 1 : 0);
                    }

                    checkNode(this.root);
                    return true;
                }

                inOrder() {
                    const res = [];
                    function traverse(n) {
                        if (!n) return;
                        traverse(n.left);
                        res.push(n.key);
                        traverse(n.right);
                    }
                    traverse(this.root);
                    return res;
                }
            }

            const tree = new RedBlackTree();
            const keys = [20, 15, 25, 10, 5, 1, 30, 28, 35, 18, 12];
            keys.forEach(k => tree.insert(k));

            console.log("Keys inserted:", keys.length);
            const verified = tree.verify();
            console.log("RB-Tree properties verified:", verified);
            const sorted = tree.inOrder();
            console.log("In-order traversal:", sorted.join(", "));
            "红黑树测试全部通过";
        """.trimIndent()

        val result = JsRuntime.execute(rbTreeScript)
        assertTrue("Execution should succeed: ${result.error}", result.success)
        assertEquals("红黑树测试全部通过", result.result)
        assertTrue(result.stdout.contains("RB-Tree properties verified: true"))
        assertTrue(result.stdout.contains("In-order traversal: 1, 5, 10, 12, 15, 18, 20, 25, 28, 30, 35"))
    }

    @Test
    fun testInfiniteLoopTimeoutProtection() {
        val infiniteScript = "while(true) {}"
        val result = JsRuntime.execute(infiniteScript, timeoutMs = 500L)
        assertFalse("Infinite loop must fail", result.success)
        assertTrue("Should report timeout or instruction quota error",
            result.error?.contains("超时") == true || result.error?.contains("配额") == true
        )
    }

    @Test
    fun testSandboxBlocksJavaAccess() {
        val hackScript = """
            try {
                java.lang.System.exit(0);
                "escaped";
            } catch(e) {
                "blocked: " + e.message;
            }
        """.trimIndent()
        val result = JsRuntime.execute(hackScript)
        assertTrue(result.success)
        assertTrue(result.result?.contains("blocked") == true || result.result?.contains("not defined") == true)
    }
}
