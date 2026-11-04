from rag import rag_chain
import csv


test_questions = [

    {
        "question": "What is the Transformer architecture?",
        "expected_answer": (
            "The Transformer is an architecture based entirely "
            "on attention mechanisms, dispensing with recurrence "
            "and convolutions."
        )
    },

    {
        "question": "How many encoder and decoder layers are used?",
        "expected_answer": (
            "The Transformer uses six encoder layers "
            "and six decoder layers."
        )
    },

    {
        "question": "What is the dimensionality of the model?",
        "expected_answer": (
            "The base Transformer model uses a model "
            "dimension of 512."
        )
    },

    {
        "question": "How many attention heads are used?",
        "expected_answer": (
            "The base Transformer model uses eight "
            "attention heads."
        )
    },

    {
        "question": "What is positional encoding?",
        "expected_answer": (
            "Positional encoding provides information "
            "about the positions of tokens in a sequence."
        )
    }

]


def run_evaluation():

    results = []

    for index, test in enumerate(test_questions, start=1):

        question = test["question"]
        expected_answer = test["expected_answer"]

        print("\n" + "=" * 60)
        print(f"TEST CASE {index}")
        print("=" * 60)

        response = rag_chain.invoke(question)

        generated_answer = response["answer"]

        documents = response["documents"]

        sources = []

        for doc in documents:

            filename = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            if isinstance(page, int):
                page = page + 1

            source = f"{filename} | Page {page}"

            if source not in sources:
                sources.append(source)

        results.append({
            "Question": question,
            "Expected Answer": expected_answer,
            "Generated Answer": generated_answer,
            "Retrieved Sources": "; ".join(sources)
        })

        print("\nQuestion:")
        print(question)

        print("\nExpected Answer:")
        print(expected_answer)

        print("\nGenerated Answer:")
        print(generated_answer)

        print("\nRetrieved Sources:")
        for source in sources:
            print(source)

    with open(
        "evaluation_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Question",
            "Expected Answer",
            "Generated Answer",
            "Retrieved Sources"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nEvaluation completed!")
    print("Results saved to evaluation_results.csv")


def test_out_of_scope_question():

    question = "Who won the FIFA World Cup in 2018?"

    print("\n" + "=" * 60)
    print("OUT-OF-SCOPE TEST")
    print("=" * 60)

    response = rag_chain.invoke(question)

    print("\nQuestion:")
    print(question)

    print("\nGenerated Answer:")
    print(response["answer"])

    print("\nRetrieved Documents:")

    for doc in response["documents"]:
        print(doc.metadata)



if __name__ == "__main__":

    run_evaluation()

    test_out_of_scope_question()            
