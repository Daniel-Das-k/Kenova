from autogen import UserProxyAgent, AssistantAgent, config_list_from_json
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import autogen

load_dotenv()

class AnswerEvaluator:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.setup_config()
        self.agents = None
    
    def setup_config(self):
        """Setup basic configurations for the agents."""
        self.gemini_config_list = config_list_from_json(
            "OAI_CONFIG_LIST.json",
            filter_dict={"model": ["gemini-2.0-flash-exp"]},
        )
        
        self.llm_config = {
            "config_list": self.gemini_config_list,
            "seed": 53,
            "temperature": 0.7,
            "timeout": 300,
        }

    def _setup_evaluation_agents(self, sample_answers):
        """Setup agents for evaluation"""
        initializer = UserProxyAgent(
            name="Initializer",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
            is_termination_msg=lambda msg: "json" in msg["content"].lower()
        )

        analyser_agent = AssistantAgent(
            name="Analyser_agent",
            llm_config=self.llm_config,
            system_message=f"""
            You are an expert in evaluating short-answer, long answer and case study based responses submitted by students. Your task is to analyze the given question, the correct answer, and the student's response. 

            The student's answer may consist of keywords or partial concepts rather than fully developed responses. Assess whether the provided keywords align with the key concepts of the correct answer and determine if the response demonstrates an understanding that could be further developed into a complete answer.

            Remember, only keywords will be provided.

            Provide detailed feedback, highlighting strengths, missing elements, and areas for improvement. If the student's response is incomplete or lacks clarity, suggest how it can be expanded. Ensure fairness and accuracy in scoring, considering partial credit where applicable.

            The provided feedback should help the student learn and improve on areas where they are struggling.(do not include The students provides)

            The response should be first person and not third person.
            Return a structured evaluation in the following JSON format:
            If the answer is partially correct for short answer(2-mark) give 0.5 mark to 1.5 mark, if the answer is partially correct for long answer(13-mark) give 4 to 11 marks, if the answer is partially correct for case study(15-mark) give 6 to 13 marks, if the answer is incorrect for short answer(2-mark) give 0 mark, if the answer is incorrect for long answer(13-mark) give 0 mark, if the answer is incorrect for case study(15-mark) give 0 mark.
            If the answer is fully correct award max marks 
            Example Response:
            {{
                "Correctness": (True/False/Partially Correct),
                "Score": (Marks obtained out of max_marks),
                "Feedback": (Detailed explanation on alignment with the correct answer, missing details, and suggestions for improvement.)
            }}

            Sample Answers for Reference:
            {sample_answers}
            """

        )

        final_agent = AssistantAgent(
            name="final_agent",
            llm_config=self.llm_config,
            system_message= f"""
                You are an expert in combining the responses obtained from the Analyser_agent. Your task is to merge all the responses into a structured format, ensuring consistency and clarity. 

                Return the combined evaluation in the following JSON format:

                Example Response:
                {{
                    "responses": [
                        {{
                            "question_number": 1,
                            
                            "Correctness": "Partially Correct",
                            "Score": 1,
                            "Feedback": "The student's answer is incorrect. The capital of France is Paris, not Berlin."
                          
                        }},
                        {{
                            "question_number": 2,
                            "Correctness": "Incorrect",
                            "Score": 0,
                            "Feedback": "The student's answer is incorrect. The largest planet in our solar system is Jupiter, not Mars."
                            }},
                    
                        {{
                            "question_number": 3,
                            "Correctness": "Incorrect",
                            "Score": 0,
                            "Feedback": "The student's answer is incorrect. The chemical symbol for gold is Au, not Ag."

                            }},
                            {{
                            "question_number": 11,
                            "Correctness": "Incorrect",
                            "Score": 6,
                            "Feedback": "The student's answer is partially correct. The chemical symbol for gold is He, not H."

                            }},
                            {{
                            "question_number": 15,
                            "Correctness": "Incorrect",
                            "Score": 9,
                            "Feedback": "The student's answer is partially correct. The chemical symbol for gold is Hg, not Hg."

                            }},
                            ...........................
                    }}
       
                Now, merge and format the responses provided by the Analyser_agent accordingly.
                """

        )

        return [initializer, analyser_agent,final_agent]

    def batch_evaluate(self, sample_answers):
        """Evaluate a batch of answers"""
        self.agents = self._setup_evaluation_agents(sample_answers)
        try:
            # Initialize group chat for evaluation
            groupchat = autogen.GroupChat(
                agents=self.agents,
                messages=[],
                max_round=3,
                speaker_selection_method="round_robin"
            )
            manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

            # Start evaluation process
            chat_result = self.agents[0].initiate_chat(
                manager,
                message=f"""Evaluate these student responses:
                {json.dumps(sample_answers, indent=2)}
                Provide a detailed evaluation for each answer following the scoring rules:
                - For 2-mark questions: 0 for incorrect, 0.5-1.5 for partially correct, 2 for fully correct
                - For 13-mark questions: 0 for incorrect, 4-11 for partially correct, 13 for fully correct
                - For 15-mark questions: 0 for incorrect, 6-13 for partially correct, 15 for fully correct
                Return the evaluation as a JSON response containing the evaluation of all questions."""
            )

            # Extract the final evaluation result
            final_result = None
            for message in chat_result.chat_history:
                if 'content' in message:
                    final_output = message['content']
            
            return final_output
            
        except Exception as e:
            print(f"Error in batch_evaluate: {str(e)}")
            return {'status': 'error', 'message': str(e)}

def main():
    # Initialize evaluator with API key
    api_key = os.getenv("GOOGLE_API_KEY")  # Get API key from environment variable
    
    # Example answers to evaluate
    sample_answers = [
                {
                    "question_number": 1,
                    "question_text": "What is the purpose of the im2col function in the context of convolutional neural networks?",
                    "answer": "The im2col function transforms the input data into a matrix, facilitating efficient convolution operations by converting the sliding window process into matrix multiplication. It expands the input data for easier processing.",
                    "user_answer": "im2col helps in convolution by transforming data.",
                    "marks": 2
                },
                {
                    "question_number": 2,
                    "question_text": "Explain the difference in how a fully connected layer and a convolution layer handle input data shape.",
                    "answer": "A fully connected layer ignores the spatial shape of input data, treating all inputs as equivalent neurons. A convolution layer, however, maintains the shape, processing data as multi-dimensional arrays, which is crucial for image data.",
                    "user_answer": "A fully connected layer treats all inputs as equivalent neurons, while a convolution layer maintains the spatial shape of input data, crucial for image data.",
                    "marks": 2
                },
                {
                    "question_number": 3,
                    "question_text": "What are the three main steps involved in implementing a pooling layer?",
                    "answer": "The three steps are: 1. Expand the input data using im2col. 2. Take the maximum value in each row of the expanded matrix. 3. Reshape the output to the appropriate dimensions.",
                    "user_answer": "The three steps in implementing a pooling layer are: 1. Expand the input data using im2col. 2. Take the maximum value in each row of the expanded matrix. 3. Reshape the output to the appropriate dimensions.",
                    "marks": 2
                },
                {
                    "question_number": 4,
                    "question_text": "What is the role of a filter (or kernel) in a convolution operation?",
                    "answer": "A filter (or kernel) is a small matrix used to extract features from an image. It slides across the input data, performing element-wise multiplication and summation to produce an output feature map.",
                    "user_answer":"I love the mam",
                    "marks": 2
                },
                {
                    "question_number": 5,
                    "question_text": "How is the output size of a convolution operation calculated, given an input size of m*m and a kernel size of n*n?",
                    "answer": "The output size of a convolution operation is calculated as (m - n + 1) * (m - n + 1). This formula determines the dimensions of the resulting feature map after convolution.",
                    "user_answer":"69",
                    "marks": 2
                },
                 {
                    "question_number": 6,
                    "question_text": "What is the purpose of padding in a convolution operation?",
                    "answer": "Padding is not explicitly mentioned in the provided context, but it is a technique used to add extra layers of pixels around the input image to control the output size and prevent information loss at the edges.",
                    "user_answer":"padding is the best",
                    "marks": 2
                },
                {
                    "question_number": 7,
                    "question_text": "What is the difference between a stride and a filter size in a convolution operation?",
                    "answer": "Filter size refers to the dimensions of the kernel used for convolution, while stride is the number of pixels the filter moves in each step across the input data. Stride affects the output size.",
                    "user_answer":"the filter size is the best",
                    "marks": 2
                },
                {
                    "question_number": 8,
                    "question_text": "What are the key components of a typical CNN architecture?",
                    "answer": "A typical CNN architecture includes convolution layers, pooling layers, and fully connected layers. These layers work together to extract features, reduce dimensionality, and make final predictions.",
                    "user_answer":"I dont know the answer",
                    "marks": 2
                },
                {
                    "question_number": 9,
                    "question_text": "What is the role of the transpose function in the context of multidimensional arrays?",
                    "answer": "The transpose function changes the order of axes in a multidimensional array. It allows for rearranging the dimensions of the array, which can be useful for various data manipulations.",
                    "user_answer":"I dont know the answer",
                    "marks": 2
                },
                {
                    "question_number": 10,
                    "question_text": "What is the purpose of the bias in a convolution operation?",
                    "answer": "Bias in a convolution operation is a fixed value added to each element of the output feature map after the filter is applied. It allows the model to learn an offset in the feature detection.",
                    "user_answer":"I dont know the answer",
                    "marks": 2
                },
                
    {
        "question_number": 11,
        "question_text": "Critically analyze the role of the `im2col` function in the implementation of both convolution and pooling layers within a Convolutional Neural Network (CNN). Discuss the advantages and potential drawbacks of using `im2col` compared to a more direct implementation of these operations. Furthermore, explain how the batch size affects the output of the `im2col` function, providing specific examples from the provided text.",
        "answer": "The `im2col` function serves as a crucial bridge between the spatial nature of convolutional and pooling operations and the matrix-based computations that are highly optimized in modern computing environments. In essence, `im2col` transforms the input data into a matrix format, enabling the use of efficient matrix multiplication routines to perform what would otherwise be a more complex sliding window operation. This transformation is particularly beneficial for convolution layers, where a filter (or kernel) is applied across the input data, and for pooling layers, where a region of the input is reduced to a single value (e.g., maximum or average). \n\n**Convolution Layer and `im2col`:**\nIn a convolution layer, the filter slides across the input data, performing element-wise multiplication and summation at each location. Without `im2col`, this would require nested loops to iterate over the input data and the filter. However, `im2col` restructures the input data such that each sliding window is represented as a row in a matrix. The filter is also reshaped into a matrix, and the convolution operation is then performed as a matrix multiplication between these two matrices. The result is then reshaped back into the output feature map. This approach leverages highly optimized matrix multiplication libraries, leading to significant performance gains, especially for large input data and filters. The text mentions that with a 7x7 input, 3 channels, and a 5x5 filter, the `im2col` function results in a matrix with 75 columns, which corresponds to the total number of elements in the filter (3 channels * 5 * 5). When the batch size is 1, the output is (9, 75), and when the batch size is 10, the output is (90, 75). This demonstrates how `im2col` expands the input data to accommodate the batch size, allowing for parallel processing of multiple input samples.\n\n**Pooling Layer and `im2col`:**\nSimilarly, in a pooling layer, `im2col` expands the input data to facilitate the pooling operation. The key difference is that pooling is independent of the channel dimension. For example, with a 2x2 pooling area, `im2col` expands the input data such that each 2x2 region is represented as a row in a matrix. The pooling operation (e.g., taking the maximum value) is then performed on each row, and the result is reshaped into the output feature map. This approach simplifies the implementation of pooling and allows for efficient computation.\n\n**Advantages of `im2col`:**\n1. **Performance:** The primary advantage of `im2col` is its ability to leverage highly optimized matrix multiplication libraries, leading to significant performance improvements compared to direct implementations using nested loops.\n2. **Simplicity:** By transforming the convolution and pooling operations into matrix multiplications, `im2col` simplifies the implementation of these layers, making the code more concise and easier to understand.\n3. **Parallelization:** The matrix multiplication operation is highly parallelizable, allowing for efficient use of modern hardware, such as GPUs.\n\n**Potential Drawbacks of `im2col`:**\n1. **Memory Overhead:** The primary drawback of `im2col` is the increased memory usage. The expanded matrix can be significantly larger than the original input data, especially for large input sizes and filter sizes. This can be a limiting factor for very large networks or when working with limited memory resources.\n2. **Computational Overhead:** While matrix multiplication is highly optimized, the `im2col` transformation itself introduces some computational overhead. This overhead can be significant for small input sizes or when the transformation is not implemented efficiently.\n\n**Impact of Batch Size:**\nThe batch size directly affects the output of the `im2col` function. As demonstrated in the text, when the batch size is 1, the output of `im2col` is (9, 75), and when the batch size is 10, the output is (90, 75). The first dimension of the output matrix is multiplied by the batch size, while the second dimension remains constant. This allows for processing multiple input samples in parallel, which is crucial for efficient training of neural networks. The `im2col` function effectively expands the input data to accommodate the batch size, enabling the parallel computation of convolution and pooling operations across multiple samples. In summary, `im2col` is a powerful technique for implementing convolution and pooling layers, but it is essential to consider its memory overhead and computational cost when designing and implementing CNNs.",
        "marks": 13,
        "user_response": "im2col, convolution, pooling, feature extraction, dimensionality reduction, batch size, matrix multiplication, performance, memory overhead, computational overhead, parallel processing"
    },
    {
        "question_number": 12,
        "question_text": "Compare and contrast the operational characteristics of convolution layers and pooling layers within a CNN. Specifically, discuss their respective roles in feature extraction and dimensionality reduction. Furthermore, analyze how the parameters of these layers (e.g., filter size, stride, pooling size) influence the output feature maps and the overall performance of the network. Provide examples from the text to support your analysis.",
        "answer": "Convolutional and pooling layers are fundamental building blocks of Convolutional Neural Networks (CNNs), each playing a distinct yet complementary role in the network's ability to learn hierarchical features from input data. While both layers process input data using a sliding window approach, their objectives and operational characteristics differ significantly.\n\n**Convolution Layers:**\nConvolution layers are primarily responsible for feature extraction. They achieve this by applying a set of learnable filters (or kernels) to the input data. Each filter slides across the input, performing element-wise multiplication and summation to produce an output feature map. The filters are designed to detect specific patterns or features in the input data, such as edges, corners, or textures. The output of a convolution layer is a set of feature maps, each corresponding to a different filter. The text mentions that the convolution operation is equivalent to the filter operation in image processing, and it uses the word 'kernel' for the term 'filter'. The text also provides examples of standard filters like the Sobel filter and the Scharr filter, which are used for edge detection. The convolution operation maintains the spatial structure of the input data, allowing the network to learn spatially localized features. The text also highlights that the input data for a convolution layer is called an input feature map, while the output data is called an output feature map.\n\n**Pooling Layers:**\nPooling layers, on the other hand, are primarily responsible for dimensionality reduction and spatial invariance. They reduce the spatial size of the feature maps, which helps to reduce the computational cost and the number of parameters in the network. Pooling layers also make the network more robust to small variations in the input data. The most common pooling operations are max pooling and average pooling. Max pooling selects the maximum value within each pooling region, while average pooling calculates the average value. The text describes the implementation of a pooling layer using `im2col` to expand the input data, followed by taking the maximum value in each row and reshaping the output. Unlike convolution layers, pooling layers do not have learnable parameters. They simply perform a fixed operation on the input data.\n\n**Comparison:**\n| Feature | Convolution Layer | Pooling Layer |\n|---|---|---|\n| **Primary Role** | Feature Extraction | Dimensionality Reduction, Spatial Invariance |\n| **Operation** | Applies learnable filters to input data | Reduces spatial size of feature maps |\n| **Parameters** | Filter weights, biases | None (fixed operation) |\n| **Output** | Set of feature maps | Reduced-size feature maps |\n| **Spatial Structure** | Maintains spatial structure | Reduces spatial structure |\n\n**Influence of Parameters:**\n1. **Filter Size:** The filter size in a convolution layer determines the spatial extent of the features that the filter can detect. Smaller filters can detect fine-grained features, while larger filters can detect more coarse-grained features. The text mentions that a 3x3 kernel matrix is very common. The choice of filter size depends on the complexity of the input data and the desired level of detail in the features.\n2. **Stride:** The stride in a convolution layer determines how much the filter moves across the input data in each step. A larger stride results in a smaller output feature map and reduces the computational cost. However, it can also lead to a loss of information. The text mentions that the convolution operation is applied while the filter window is shifted at a fixed interval. The stride is a hyperparameter that needs to be tuned based on the specific task.\n3. **Pooling Size:** The pooling size in a pooling layer determines the size of the region over which the pooling operation is performed. A larger pooling size results in a greater reduction in the spatial size of the feature maps. However, it can also lead to a loss of information. The text mentions that the target pooling area is 2x2. The choice of pooling size depends on the desired level of dimensionality reduction and the robustness to spatial variations.\n\n**Examples from the Text:**\nThe text provides several examples that illustrate the concepts discussed above. For instance, the text mentions that the input size is (4, 4), the filter size is (3, 3), and the output size is (2, 2). This example demonstrates how the filter size and the input size affect the output size of a convolution layer. The text also mentions that the output size of a pooling layer can be reduced using a 2x2 pooling window with a stride of 2, leading to a reduction in the spatial size of the feature map.\n\n**Conclusion:**\nIn summary, convolution layers are responsible for extracting features from the input data, while pooling layers perform dimensionality reduction and improve the network's robustness. The parameters of these layers, such as filter size, stride, and pooling size, significantly influence the output feature maps and the overall performance of the network. A well-designed CNN carefully balances these parameters to achieve optimal performance on the task at hand.",
        "marks": 13,
        "user_response": "convolution, pooling, feature extraction, dimensionality reduction, filter size, stride, pooling size, max pooling, average pooling, spatial invariance"
    },
    {
        "question_number": 13,
        "question_text": "Explain the concept of overfitting in machine learning. Discuss the causes, signs, and potential methods to prevent or mitigate overfitting in a model. Provide specific examples from the text.",
        "answer": "Overfitting is a common issue in machine learning, where a model learns the training data too well, including the noise and outliers, and fails to generalize effectively to new, unseen data. In other words, an overfitted model performs well on the training set but poorly on the test set or any new data, as it has essentially memorized the training data instead of learning general patterns. Overfitting can occur when the model is too complex relative to the amount of training data available, or when the model is trained for too many epochs without adequate regularization.\n\n**Causes of Overfitting:**\n1. **Complexity of the Model:** Overfitting is more likely when the model has too many parameters relative to the size of the training dataset. A model with excessive capacity can fit the training data very closely, including noise and outliers, but it will struggle to generalize to new data.\n2. **Insufficient Training Data:** If the training dataset is too small, the model may learn the specific idiosyncrasies of the data, leading to overfitting. A small dataset may not provide enough variation for the model to learn general patterns, causing it to memorize the data.\n3. **Training for Too Many Epochs:** Training a model for too many epochs can result in overfitting, especially if the model starts to fit the noise in the data. If the model's performance on the training set continues to improve while the performance on the test set deteriorates, this is a strong indicator of overfitting.\n\n**Signs of Overfitting:**\n1. **Good Performance on Training Set, Poor Performance on Test Set:** A classic sign of overfitting is when the model performs well on the training data but poorly on the test data or new data. This indicates that the model has memorized the training data rather than learned generalizable patterns.\n2. **Large Gap Between Training and Test Error:** When there is a significant gap between the error on the training set and the error on the test set, it suggests overfitting. The model may have over-learned the specific characteristics of the training data that do not apply to new data.\n\n**Methods to Prevent or Mitigate Overfitting:**\n1. **Simplifying the Model:** One of the simplest ways to reduce overfitting is by using a simpler model with fewer parameters. This reduces the model's capacity and forces it to learn more general patterns from the data.\n2. **Regularization:** Techniques such as L1 and L2 regularization add a penalty term to the loss function, discouraging the model from fitting the training data too closely. Regularization helps to prevent overfitting by reducing the complexity of the model.\n3. **Cross-Validation:** Cross-validation is a technique where the dataset is split into several subsets, and the model is trained on different combinations of these subsets while being validated on the remaining data. This helps to ensure that the model generalizes well and is not just memorizing the training data.\n4. **Data Augmentation:** Data augmentation involves artificially increasing the size of the training dataset by applying transformations such as rotations, shifts, or flips to the original data. This introduces more variation into the training data, helping the model learn more robust features.\n5. **Early Stopping:** Early stopping involves monitoring the performance of the model on the validation set and stopping the training process when the performance starts to degrade. This prevents the model from continuing to learn noise in the data and helps to avoid overfitting.\n\n**Examples from the Text:**\nThe text provides several examples of overfitting, such as when the model continues to improve on the training set but starts to perform worse on the test set. The text also mentions that overfitting can occur when a model with too many parameters is trained on a small dataset. For instance, the text provides an example where a model with a high number of parameters was able to achieve very low training error but had poor performance on the test set.\n\n**Conclusion:**\nIn summary, overfitting occurs when a model learns the training data too well and fails to generalize to new data. It can be caused by factors such as model complexity, insufficient data, and excessive training. The signs of overfitting include a large gap between training and test performance. Methods such as simplifying the model, regularization, cross-validation, data augmentation, and early stopping can help prevent or mitigate overfitting.",
        "marks": 13,
        "user_response": "overfitting, model complexity, insufficient data, regularization, cross-validation, data augmentation, early stopping, training error, test error, generalization"
    },
    {
            "section_title": "Case Study",
            "total_marks": 15,
            "background": "A small e-commerce startup, 'GadgetHub,' is experiencing rapid growth. They sell a variety of electronic gadgets and accessories. Initially, they managed their inventory and sales using a simple spreadsheet. However, with increasing sales volume and product variety, their current system is becoming inefficient and prone to errors. They are facing challenges in tracking inventory levels, managing customer orders, and generating sales reports. The manual process is also causing delays in order fulfillment and customer service. They have a small team with limited technical expertise and budget.",
            "problem_statement": "GadgetHub needs to implement a more efficient and scalable system to manage their inventory, sales, and customer orders. They require a solution that is easy to use, cost-effective, and can be implemented quickly with their limited resources.",
            "supporting_data": [
                "Current monthly sales: 500 orders",
                "Number of products: 200",
                "Average order value: $50",
                "Current inventory tracking method: Spreadsheet",
                "Customer order management: Manual process"
            ],
            "questions": [
                {
                    "question_number": 1,
                    "question_text": "Analyze the current challenges faced by GadgetHub and propose a suitable system to address their needs. Justify your choice of system, considering their limited resources and technical expertise. Discuss the key features that the proposed system should include to effectively manage their inventory, sales, and customer orders. Furthermore, outline a phased implementation plan for the proposed system, considering their limited resources and the need for minimal disruption to their operations.",
                    "marks": 15,
                    "user_response":"""Cloud-based system
                Odoo or Zoho Inventory
                Cost-effective, scalable, user-friendly"""
                }
            ]
    }
    ]
            
    evaluator = AnswerEvaluator()
    results = evaluator.batch_evaluate(sample_answers)
 
    
    

if __name__ == "__main__":
    main()