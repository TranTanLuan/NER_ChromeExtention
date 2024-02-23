# Chrome extension
## Introduction
Chrome extension, it reads the content of the current page and sent it to a server (django python backend), that server calls the NER model to predict entities and send the results back to extension

## Directory
- manifest.json: has some information (name, version, description), define how the extension appears in the browser’s toolbar and what happens when clicked (popup.html) 
- background.js: 
- popup.js:  
- content.js:

## Overall Workflow
- when click button
- content.js read the content of the webpage and send them to popup.js
- popup.js receive, send them to background.js
- background.js receive, to send them to server, and receive the response from server, send that response to popup.js
- popup.js receive, send them back to background.js 
- background.js receive, send them to content.js
- content.js receive, process, display results on webpage