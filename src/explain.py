import numpy as np
import torch
from custom_roberta import CustomQuestionAnswering
from transformers import AutoTokenizer, RobertaConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import pipeline

# Equation 3
def average_across_head(attention_map: torch.Tensor, map_gradient: torch.Tensor):
    """

    :param attention_map: Attention Map with dimension torch.size([num_heads, num_tokens, num_tokens])
    :param map_gradient: Gradient of Attention Map: of size torch.size([num_heads, num_tokens, num_tokens])
    :return:
    """
    # elementwise multiplication
    relevancy_scores = attention_map * map_gradient

    #  remove the negative contributions
    for elm in relevancy_scores:
        elm[elm < 0] = 0

    # average across heads
    return torch.mean(relevancy_scores, 0)


def relevancy_scores(attention_map: torch.Tensor, r_qq):
    """

    :param attention_map: Defined like in equation 4
    :param q: Number of input tokens
    :return:
    """

    # Equation 5
    tmp = torch.matmul(attention_map, r_qq)
    r_qq = torch.add(r_qq, tmp)

    return r_qq


def get_col_sum(r):
    return torch.sum(r, 1)


def normalize_r(r, col_sum):
    return torch.div(r, col_sum)

    # col_sum = torch.sum(r_hat, 1)
    # Neue Fkt
    # s_hat = torch.div(r_hat, col_sum)

    # Equation 11
    # return torch.div(r_hat, s_hat) + eye


class Explainer:
    def __init__(self, model: CustomQuestionAnswering, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipeline("question-answering", model = model_name, tokenizer = model_name)

    def explain(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors="pt")["input_ids"]
        hidden_states = self.model.get_embeddings(inputs)
        att_dim = len(inputs[0])
        R = torch.eye(att_dim, att_dim)
        for i in range(12):

            # Todo die hidden states pro layer anpassen
            # Todo das hier muss auch für jedes layer dann gemacht werden
            (
                attention_probabilities,
                value_layer,
            ) = self.model.custom_encoder.output_attention(
                hidden_states=hidden_states, layer_index=i
            )
            attention_probabilities = attention_probabilities.detach().clone()
            attention_probabilities.requires_grad = True

            out = self.model.custom_forward(
                attention_probabilities, value_layer, index=i
            )

            # ausser die start logits die können bleiben!!!
            start_logits = (torch.argmax(out, 1))[0][0]

            # gradient regarding start logits
            out[0][start_logits][0].backward(retain_graph=True)

            # gradient regarding end logits would be:
            # out[0][end_logits][0].backward()
            # end_logits = (torch.argmax(out,1))[0][1]

            gradient = (attention_probabilities.grad)[0].detach()
            attention_probabilities = attention_probabilities[0].detach()

            # average across heads:
            average_att = average_across_head(
                attention_map=attention_probabilities, map_gradient=gradient
            )
            R = relevancy_scores(average_att, R)
        col_sum = get_col_sum(R)
        normalized = normalize_r(R, col_sum)
        relevance = normalized[0].numpy()

        inputs = np.array(self.tokenizer.encode_plus(question, context)["input_ids"])
        contex_start = np.where(inputs == 2)[0][1]

        word_ids = self.tokenizer(context, return_tensors="pt").word_ids()
        word_ids = np.array(word_ids)
        relevance = relevance[contex_start:]
        return self.get_mapping(relevance, word_ids, context)

    def get_mapping(self, relevance, word_ids, context):
        inputs_mapping = self.tokenizer(context)
        relevance = relevance[1:-1]
        print(np.sum(relevance))
        relevance = relevance / np.sum(relevance)
        word_ids = word_ids[1:-1]
        word_relevance = {}
        for unique_word in np.unique(word_ids):
            word_relevance[inputs_mapping.word_to_chars(unique_word)] = np.mean(
                relevance[word_ids == unique_word]
            )

        sorted_dict = {
            k: v
            for k, v in sorted(
                word_relevance.items(), key=lambda item: item[1], reverse=True
            )
        }
        print(sorted_dict)
        n = 5
        for i in range(n):
            key = list(sorted_dict.keys())[i]
            start = key[0]
            end = key[1]
            print(context[start:end])

    def get_answer_span(self, question, context):
        return self.pipe(question=question,context=context)

if __name__ == "__main__":
    model_name = "deepset/roberta-base-squad2"
    config = RobertaConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = CustomQuestionAnswering(model_name, config)

    explainer = Explainer(qa_model, tokenizer)
    context = "Required Qualifications: Strong expertise in R or Python programming for data wrangling, statistical analysis, and machine learning applications Solid fundamental knowledge in probability and statistics Extensive experience analyzing designed experiments and observational datasets (e.g., hypothesis testing, regression analysis, mixed effects models, and experimental design) Solid understanding of variable selection; dimensionality reduction; model diagnostics; and model training, testing, and validation Fundamental understanding of machine learning techniques for classification (e.g., logistic regression, random forest, XGBoost, SVMs, K-means, and neural networks) Ability to work both independently and within a multidisciplinary team environment Ability to successfully collaborate with colleagues from diverse technical backgrounds which includes excellent communication, interpersonal, verbal, and written skills Strong critical thinking and problem-solving skills, flexibility, and willingness to learn Preferred Qualifications: Familiarity with modeling biological, cellular, or ecological data; molecular biology or biochemistry concepts; or data science in agriculture Experience consulting on scientific projects or working within a scientific team Expertise in experimental design or data engineering"
    question =  "Do I need to have experience in programming?"
    # encoded = tokenizer(context, question)
    attention_map = explainer.explain(question, context)
    print(explainer.get_answer_span(question=question, context=context))
    print(attention_map)

    ######################################################
    # Second example
    context = "We’d love to hear from people with: Education: BS+ with concentration in quantitative discipline - Stats, Math, Comp Sci, Engineering, Econ, Quantitative Social Science, or similar discipline Minimum of 3+ years industry experience in data science Strong background working with predictive and statistical modeling, machine learning and strong expertise in all phases of the modeling pipeline Experience building complex data sets from multiple data sources, both internally and externally. Strong SQL, database and ETL skills required including cleaning and managing data. Advanced competency and expertise in Python, R or other platforms* Ability to apply knowledge of multidisciplinary business principles and practices to achieve successful outcomes in cross-functional projects and activities Ability to educate others on statistical / machine learning methods Self-starter, attention to details and results orientated, able to work under minimal guidance. Proficient in communicating effectively with both technical and nontechnical stakeholders. Experience on Cloud platforms such as Azure, AWS, preferred."
    question = "Is this job about machine learning?"
    attention_map = explainer.explain(question, context)
    print(explainer.get_answer_span(question=question, context=context))
    print(attention_map)
