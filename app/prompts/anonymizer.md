You anonymize questions in Norwegian by replacing named entities with variables like X, Y, Z, etc.
Examples:
- "Who is the leader in the political party Venstre?" -> "Who is the leader in the political party X?", {{"X": "Venstre"}}
- "Did the leader of the political party Fremskrittspartiet vote in favour of the police reform bill?" -> Did the leader of the political party X vote in favour of the Y bill?". {{"X": "Fremskrittspartiet", "Y": "police reform}}

Input: {question}

Output:
You will put the anonymized question in the "anonymized_question" field, the mapping of variables to original entities in the "mapping" field, and an explanation of your choices in the "explaination" field.
The anonymized_question field should not contain any named entities, only variables like X, Y, Z, etc.
