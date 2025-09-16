# Prompt: Convert Markdown Questions to JSON

You are given a multiple-choice question from a Markdown file. Convert it into the following JSON format:

```json
{
  "_id": { "$oid": "<generate a random ObjectId-like string>" },
  "question": "<the main question text>",
  "one": "<first option>",
  "two": "<second option>",
  "three": "<third option>",
  "four": "<fourth option>",
  "correct": "<the correct option key: one | two | three | four>",
  "category": "<category of the question, if provided>",
  "QuestionPic": "<image URL if provided, else empty string>",
  "__v": { "$numberInt": "0" }
}
```

## Rules:
1. Extract the **question text** from the Markdown.  
2. Map each choice to `"one"`, `"two"`, `"three"`, `"four"` in order.  
3. If the correct answer is specified in the Markdown, put its key (`"one"`, `"two"`, etc.) in `"correct"`.  
4. If there’s a category, include it; otherwise, leave it empty.  
5. If there’s an image, put the URL in `"QuestionPic"`; else use an empty string.  
6. Always set `"__v":{"$numberInt":"0"}`.  
7. Generate a random 24-character hex string for `"$oid"`.  

## Example Input (Markdown)
```md
### Category: AC Repair
**Question:** Do you know AC gas charging and repairing/ servicing?

- Yes ✅
- No
- Willing to Learn
- Don't know about it and not willing to learn as well.
```

## Example Output (JSON)
```json
{
  "_id": { "$oid": "67758325bbe41a005ac8a2bd" },
  "question": "Do you know AC gas charging and repairing/ servicing?",
  "one": "Yes",
  "two": "No",
  "three": "Willing to Learn",
  "four": "Don't know about it and not willing to learn as well.",
  "correct": "one",
  "category": "AC Repair",
  "QuestionPic": "",
  "__v": { "$numberInt": "0" }
}
```