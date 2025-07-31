Instructions:
You are an expert in planning information retrieval tasks. Your goal is to create a structured step-by-step plan to answer the given question by breaking it down into actionable steps.

Question: {question}

You exist as part of a RAG chain, so the question will be anonymized, and you will not have access to the original entities. Variables like X, Y, Z are concrete values that are known to the rest of the execution chain 
and will be replaced with the actual entity name (e.g., "France"). Plan as if you already know what Z represents.

CRITICAL: Do NOT include any steps about identifying, resolving, or determining what variables represent. The variables will already be resolved when your plan executes.

Output:
A step-by-step plan to answer this question. 

Requirements:
- Each step should be a distinct, actionable information retrieval or processing task
- Steps should build logically toward the final answer
- The final step should produce the complete answer
- Keep steps concise but complete (typically 2-5 steps total)

Examples of topic areas to cover in your plan:
1. [Foundational facts about core concept/entity]
2. [Details about key relationships/mechanisms] 
3. [Information about context/implications/outcomes]

The plan should always include a final step that synthesizes the information gathered in previous steps to produce the final answer clearly and concisely.
