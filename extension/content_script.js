// CommentScanner: extracts visible comments from supported platforms
class CommentScanner {
    getComments() {
        const fbComments = Array.from(document.querySelectorAll("div[dir='auto']"));
        const ytSpans = Array.from(document.querySelectorAll("yt-attributed-string#content-text span"));
        return [...fbComments, ...ytSpans];
    }
}

// TextFilter: filters out short or non-Romanized Sinhala content
class TextFilter {
    static isValid(text) {
        if (!text || text.length < 3) return false;
        const hasSinhala = /[\u0D80-\u0DFF]/.test(text);
        const hasLatin = /[A-Za-z]/.test(text);
        return hasLatin || !hasSinhala;
    }
}

// Highlighter: applies styling to detected hate comments
class Highlighter {
    static highlight(element) {
        element.style.backgroundColor = "yellow";
        element.style.border = "1px solid red";
    }
}

// RomsiClient: communicates with the backend API
class RomsiClient {
    static async detectHate(text) {
        return new Promise((resolve, reject) => {
            chrome.runtime.sendMessage({ type: "detectHate", text }, (response) => {
                if (chrome.runtime.lastError) {
                    reject(chrome.runtime.lastError.message);
                } else {
                    resolve(response);
                }
            });
        });
    }
}

// RomsiExtensionController: main controller class
class RomsiExtensionController {
    constructor() {
        this.scanner = new CommentScanner();
    }

    async scanAndDetect() {
        const comments = this.scanner.getComments();

        for (const node of comments) {
            const text = node.innerText?.trim();
            if (node.dataset.checked === "true" || !TextFilter.isValid(text)) continue;
            node.dataset.checked = "true";

            try {
                const response = await RomsiClient.detectHate(text);
                if (response?.label === "hate") {
                    console.log("Hate detected:", text, "(Confidence:", response.confidence, ")");
                    Highlighter.highlight(node);
                }
            } catch (err) {
                console.error("Error detecting hate:", err);
            }
        }
    }

    run() {
        this.scanAndDetect();
        const observer = new MutationObserver(() => this.scanAndDetect());
        observer.observe(document.body, { childList: true, subtree: true });
        setInterval(() => this.scanAndDetect(), 5000);
    }
}

// Instantiate and run the extension
const controller = new RomsiExtensionController();
controller.run();
