class RomsiAPIHandler {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.listenForMessages();
    }

    listenForMessages() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            if (message.type === "detectHate") {
                this.detectHate(message.text)
                    .then(result => sendResponse(result))
                    .catch(error => {
                        console.error("API error:", error);
                        sendResponse({ error: error.toString() });
                    });
                return true; // Keep the message channel open for async response
            }
        });
    }

    async detectHate(text) {
        const response = await fetch(this.apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ texts: [text] })
        });

        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }

        const data = await response.json();
        return data.predictions[0];
    }
}

// Initialize the background API handler
const API_URL = "https://romsi-api.fly.dev/predict";
const romsiHandler = new RomsiAPIHandler(API_URL);
