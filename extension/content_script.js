console.log("✅ Romsi extension content script loaded:", window.location.href);

function containsSinhala(text) {
    return /[\u0D80-\u0DFF]/.test(text);
}

function containsLatin(text) {
    return /[A-Za-z]/.test(text);
}

function highlight(element) {
    element.style.backgroundColor = "yellow";
    element.style.border = "1px solid red";
}

function scanAndDetect() {
    // Facebook: visible comments
    const fbComments = Array.from(document.querySelectorAll("div[dir='auto']"));

    // YouTube: extract comment from span inside yt-attributed-string
    const ytSpans = Array.from(document.querySelectorAll("yt-attributed-string#content-text span"));

    const allComments = [...fbComments, ...ytSpans];

    allComments.forEach(node => {
        const text = node.innerText?.trim();
        if (!text || text.length < 3) return;

        if (node.dataset.checked === "true") return;
        node.dataset.checked = "true";

        const hasSinhala = containsSinhala(text);
        const hasLatin = containsLatin(text);

        if (hasSinhala && !hasLatin) {
            console.log("Skipped pure Sinhala:", text);
            return;
        }

        console.log("Checking text:", text);

        try {
            chrome.runtime.sendMessage({ type: "detectHate", text }, (response) => {
                if (chrome.runtime.lastError) {
                    console.error("Failed to send message:", chrome.runtime.lastError.message);
                    return;
                }

                if (response && response.label === "hate") {
                    console.log("Hate detected:", text, "(Confidence:", response.confidence, ")");
                    highlight(node);
                }
            });
        } catch (err) {
            console.error("Exception sending message:", err);
        }
    });
}

// Run immediately
scanAndDetect();

// Observe dynamic content
const observer = new MutationObserver(() => scanAndDetect());
observer.observe(document.body, { childList: true, subtree: true });

// Fallback: run every 5 seconds
setInterval(scanAndDetect, 5000);
