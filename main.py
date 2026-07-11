from rag.retriever import Retriever

retriever = Retriever()

query = "Patient P3 is urgent and caregiver S2 is delayed."

results = retriever.retrieve(query)

print("\n===== RETRIEVER RESULTS =====\n")

for result in results:

    print(f"Score : {result['score']}")
    print(f"ID : {result['document']['id']}")
    print(f"Category : {result['document']['category']}")
    print(f"Description : {result['document']['description']}")
    print("-"*50)