paragraphs = document.querySelectorAll('p');
out_str = ""

count = 0
paragraphs.forEach((paragraph) => {
    // Get original text content
    originalText = paragraph.textContent;
    out_str = out_str + "[===]" + originalText
    count += 1
});
console.log("count: ", count) // number of <p> tags
chrome.runtime.sendMessage({ action: "getText", text: out_str });

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
      
      if (request.action === "sendResult2Content") {
        console.log("content");
        list_str = request.message;
        i = 0;
        paragraphs.forEach((paragraph) => {
          newText = list_str[i];
          i += 1;
          // Update paragraph text content
          paragraph.innerHTML = newText;
        });
        }
});