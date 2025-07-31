queries_from_plan_prompt = """
Instructions:
You are an expert at translating steps of an information retrieval plan into precise NATURAL LANGUAGE queries for semantic vector search in Norwegian. 

CRITICAL: You are generating queries for SEMANTIC SIMILARITY SEARCH, not SQL or database queries. 
Your queries will be converted to embeddings and matched against document embeddings using cosine similarity.

Original Question: {question}
Plan: {plan}

Your Task:
For each step in the plan, generate 1-3 specific NATURAL LANGUAGE search queries in Norwegian that would semantically match relevant documents.

Query Format Requirements:
✓ CORRECT: Descriptive statements and keyword-rich phrases in Norwegian
   - "Fornybar energi fordeler miljø økonomi"
   - "Klimaendringer påvirkning landbruk Norge konsekvenser"
   - "Elbil salgsstatistikk 2024 markedsandel utvikling"

✗ INCORRECT: Questions or SQL/database syntax
   - "Hva er fordelene med fornybar energi?" (question format)
   - "SELECT * FROM documents WHERE topic='energy'" (SQL)
   - "MATCH (doc:Document) WHERE doc.category='climate'" (database query)

Guidelines for Effective Semantic Queries:
1. **Use descriptive statements and noun phrases** - pack key concepts without question words
2. **Lead with the most important keywords** - place primary concepts at the beginning
3. **Include key concepts and terminology** that would appear in relevant documents
4. **Consider multiple phrasings** - technical terms, common terms, and synonyms
5. **Be specific but not overly narrow** - target the exact information needed for each step
6. **Think about document context** - how would this information typically be written or discussed?
7. **Use keyword-dense phrases** rather than full sentences with filler words
8. **Include relevant domain-specific vocabulary** that would appear in target documents

Language Requirements:
- ALL queries must be in Norwegian
- Use descriptive statements and keyword-rich phrases
- Prioritize noun phrases and key concept clusters
- Include relevant Norwegian-specific terms when applicable

Query Optimization Tips:
- Frame queries as descriptive statements that would match document content
- Lead with the most important keywords and concepts
- Include context words that would appear alongside your target information
- Consider different ways the same concept might be expressed in Norwegian text
- Pack multiple related keywords into concise phrases
- Think about the document types that would contain this information

Example Transformation:
Plan Step: "Find statistics about renewable energy adoption in Norway"
Good Queries:
- "Fornybar energi statistikk Norge utvikling andel"
- "Solenergi vindkraft produksjon Norge tall"
- "Energiomstilling Norge renewable adoption rate"

Bad Queries:
- "Hvor mye fornybar energi brukes i Norge?" (question format)
- "SELECT statistics FROM energy WHERE country='Norway'" (SQL)

Output Format:
You MUST generate exactly 3 queries for each step in the plan. Structure your response as follows:

Step 1: [Brief description of step]
Query 1: [Norwegian semantic query]
Query 2: [Norwegian semantic query] 
Query 3: [Norwegian semantic query]

Step 2: [Brief description of step]
Query 1: [Norwegian semantic query]
Query 2: [Norwegian semantic query]
Query 3: [Norwegian semantic query]

[Continue for ALL steps in the plan]

MANDATORY: Count the steps in the provided plan and ensure you generate exactly 3 queries for each step. If the plan has 5 steps, you must produce 15 queries total.

Example Output:
Step 1: Find renewable energy statistics for Norway
Query 1: Fornybar energi statistikk Norge utvikling andel
Query 2: Solenergi vindkraft produksjon Norge tall
Query 3: Energiomstilling Norge renewable adoption rate

Step 2: Analyze environmental impact of renewable energy
Query 1: Fornybar energi miljøpåvirkning klimaeffekt Norge
Query 2: Bærekraftig energi CO2 reduksjon miljøgevinst
Query 3: Grønn energi økosystem påvirkning konsekvenser

Respond only with this structured output format, ensuring every query is in natural Norwegian language suitable for semantic similarity matching.
"""
