var serverhost = 'http://127.0.0.1:8000'; //defines the URL of the backend server the extension is interacting with

    chrome.runtime.onMessage.addListener( //listens for messages from other parts of the extension
        function(request, sender, sendResponse) {
          
          if (request.action === "sendTextToBackend") {
            //url: serverhost + endpoint + text (encoded for URL safety)
            var url = serverhost + '/process_text_from_extension/?text='+ encodeURIComponent(request.text) ;
			
            fetch(url) //A fetch request
            .then(response => response.json())
            // A response is sent back to the sender, containing a farewell property with the received JSON data
            .then(response => sendResponse({farewell: response}))
            .catch(error => console.log(error))
              
            return true;  // Will respond asynchronously.
            }
    });

    //listens for messages with the action sendResult
    chrome.runtime.onMessage.addListener(
      function(request, sender, sendResponse) {
        
        if (request.action === "sendResult") {
          console.log("background")
          chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, { action: "sendResult2Content", message: request.text });
          });
        }
  });