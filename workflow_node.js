export const generateResponse = async (userResponse, currentWorkflowQueue) => {
  try {
    const llamaResponse = await axios({
      method: "POST",
      url: "http://127.0.0.1:11434/api/generate", // Llama API URL
      headers: { "Content-Type": "application/json" },
      data: {
        model: "llama3.2",
        prompt: userResponse,
      },
      responseType: "stream", // Tell Axios to return a stream
    });

    const stream = llamaResponse.data;
    const result = await new Promise((resolve, reject) => {
      let fullResponse = "";

      // Read the stream data
      stream.on("data", (chunk) => {
        const chunkStr = chunk.toString();
        const lines = chunkStr.split("\n");

        lines.forEach((line) => {
          if (line.trim()) {
            try {
              const parsed = JSON.parse(line);
              if (parsed.response) {
                fullResponse += parsed.response; // Accumulate the response
              }
              if (parsed.done) {
                resolve(fullResponse); // Resolve with the final response when done
              }
            } catch (e) {
              console.error("Error parsing JSON:", e);
              reject(e); // Reject the promise if parsing fails
            }
          }
        });
      });

      // Handle stream end event
      stream.on("end", () => {
        console.log('Stream finished but "done" not reached.');
        resolve(fullResponse); // Resolve in case "done" wasn't sent
      });

      // Handle any errors in the stream
      stream.on("error", (err) => {
        console.error("Stream error:", err);
        reject(err); // Reject the promise on error
      });
    });

    currentWorkflowQueue.currentNodeId = nextNodeDetails?.nodeId ?? nextNodeId;
    await currentWorkflowQueue.save();

    console.log("Final Response:", result);
    return result;
    // console.log("llamaResponse/n", stream);
    // return "false";
  } catch (err) {
    console.error("Error in Llama", JSON.stringify(err));
    return "We are unable to process this request at the moment, please try after some time";
  }
};
