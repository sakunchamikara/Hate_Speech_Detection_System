
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "detectHate") {
        fetch("https://romsi-api.fly.dev/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texts: [message.text] })
        })
        .then(res => res.json())
        .then(data => sendResponse(data.predictions[0]))
        .catch(err => {
            console.error("API error:", err);
            sendResponse({ error: err.toString() });
        });
        return true;
    }
});
