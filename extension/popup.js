document.getElementById("getTextButton").addEventListener("click", async () => { //when button clicked
  try {
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    await chrome.scripting.executeScript({
      target: { tabId: activeTab.id },
      files: ["content.js"]
    });

    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === "getText") {
        console.log("popup");
        out_result = "";
        chrome.runtime.sendMessage({ action: "sendTextToBackend", text: request.text },
          function(response) {
            result = response.farewell;
            // document.getElementById("textOutput").textContent = result.summary;
            out_result = result.summary;
            chrome.runtime.sendMessage({ action: "sendResult", text: out_result });
        });
      }
    });
  } catch (error) {
    console.error("Error injecting script or receiving message:", error);
    // Handle the error gracefully, e.g., display an error message to the user
  }
});