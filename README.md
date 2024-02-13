
# Summarization Evaluation

Elegantly automate text summarization evaluation in reference-free manner with potential hallucination detection.

![Demo 1](assets/demo1.png)
![Demo 2](assets/demo2.png)

## Features

- **Easy to Use**: Simply provide a text file containing the summary to be evaluated, and the script will handle the rest.
- **Custom Metrics**: Utilize metrics such as word overlap and Jaccard similarity for in-depth content analysis.
- **Adapted ROUGE & BERTScore**: Rework traditional metrics for use in a reference-free context, focusing on the intrinsic qualities of summaries, as highlighted in [Eugene Yan's writing](https://eugeneyan.com/writing/abstractive/).
- **Extensible**: Easily add new metrics and models to the project to expand its capabilities, open an issue or a PR if you want to add something.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Muhtasham/summarization-eval
cd summarization-eval
pip install -r requirements.txt
```

### Usage

We run the notebook for the exp purpose and it is the main entry point for this exp.

exectution flow: This is the function that gets called from the notebook. which reuqired two string arg and one optional arg:

process_llm_results(model_summary, results_csv)

model_summary : srt-> This is the summary generted by the model
results_csv: str -> This is the data by which the summary was generated.
llm_name: str: optionl

**Note**: You will need to have an OpenAI API key set up in your environment to run the script.

## Contributing

Contributions to enhance and expand this project are welcome. Please see the `CONTRIBUTING.md` file for guidelines on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
