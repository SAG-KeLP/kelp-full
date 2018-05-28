/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.examples.demo.qc;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive.LinearPassiveAggressiveClassification;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

/**
 * This class shows how to use Kelp to build a Question classifier. In the
 * example, several kernels can be used to build your own classifier. <br>
 * <br>
 * The following kernels have been implemented: <br>
 * - bow: a Linear Kernel applied to a boolean Bag-of-Word vector, where each
 * boolean dimension indicates the presence of the corresponding word in the
 * question. <br>
 * - stk: a Subset Tree Kernel [Moschitti, 2006] over the Grammatical Relation
 * Centered Tree (GRCT) representation [Croce et al(2011)] <br>
 * - ptk: a Partial Tree Kernel [Moschitti, 2006] over the Grammatical Relation
 * Centered Tree (GRCT) representation [Croce et al(2011)]<br>
 * - sptk: a Smoothed Partial Tree Kernel over the Lexically Centered Tree (LCT)
 * representation [Croce et al(2011)]<br>
 * - csptk: Compositionally Smoothed Partial Tree Kernel over the Compositional
 * Lexically Centered Tree (CLCT) representation [Annesi et al(2014)]<br>
 * 
 * 
 * A description of the tree representations can be found in [Croce et al(2011)]
 * and [Annesi et al(2014)]. <br>
 * <br>
 * <br>
 * References:<br>
 * - [Moschitti, 2006] Alessandro Moschitti. Efficient convolution kernels for
 * dependency and constituent syntactic trees. In proceeding of European
 * Conference on Machine Learning (ECML) (2006) <br>
 * <br>
 * - [Croce et al(2011)] Croce D., Moschitti A., Basili R. (2011) Structured
 * lexical similarity via convolution kernels on dependency trees. In:
 * Proceedings of EMNLP, Edinburgh, Scotland. <br>
 * <br>
 * - [Annesi et al(2014)] Paolo Annesi, Danilo Croce, and Roberto Basili. 2014.
 * Semantic compositionality in tree kernels. In Proc. of CIKM 2014, pages
 * 1029–1038, New York, NY, USA. ACM
 * 
 * @author Danilo Croce, Simone Filice, Giuseppe Castellucci
 * 
 */
public class QuestionClassificationIncrementalLearning {

	public static void main(String[] args) {
		try {
			/*
			 * Initializing the Log level
			 */
			System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "INFO");

			/*
			 * Read both training and testing dataset
			 */
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/qc/train_5500.coarse.klp.gz");
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/qc/TREC_10.coarse.klp.gz");

			/*
			 * print some statistics
			 */
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());

			JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();

			/*
			 * Instantiate the SVM learning Algorithm. This is a binary
			 * classifier that will be transparently duplicated from the
			 * Multi-class classifier
			 */
			LinearPassiveAggressiveClassification pa = new LinearPassiveAggressiveClassification();

			/*
			 * Set the kernel
			 */
			pa.setRepresentation("bow");
			/*
			 * Set the C parameter
			 */
			pa.setC(3);
			/*
			 * Enamble the fairness: in each binary classifier, the
			 * regularization parameter of the positive examples is multiplied
			 * of a coefficient that is
			 * number_of_negative_examples/number_of_positive_examples
			 */
			pa.setFairness(true);
			/*
			 * Instantiate the multi-class classifier that apply a One-vs-All
			 * schema
			 */
			OneVsAllLearning ovaLearner = new OneVsAllLearning();
			/*
			 * Use the binary classifier defined above
			 */
			ovaLearner.setBaseAlgorithm(pa);
			ovaLearner.setLabels(trainingSet.getClassificationLabels());
			/*
			 * The classifier can be serialized
			 */
			String jsonLearner = serializer.writeValueAsString(ovaLearner);
			System.out.println(jsonLearner);
			/*
			 * Learn and get the prediction function
			 */

			Dataset[] trainParts = trainingSet.splitClassDistributionInvariant(0.05f); 
			trainParts[0] = trainParts[0].getShuffledDataset();
			trainParts[1] = trainParts[1].getShuffledDataset();

			ovaLearner.learn(trainParts[0]);
			Classifier f = ovaLearner.getPredictionFunction();
			/*
			 * After small training.
			 * Classify examples and compute the accuracy, i.e., the percentage
			 * of questions that are correctly classified
			 */
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
					trainingSet.getClassificationLabels());
			for (Example e : testSet.getExamples()) {
				// Predict the class
				ClassificationOutput p = f.predict(e);
				evaluator.addCount(e, p);
			}
			System.out.println("Accuracy after first part training: " + evaluator.getAccuracy());

			/*
			 * Write the model (aka the Classifier for further use)
			 */
			String classifierJson = serializer.writeValueAsString(f);
			//System.out.println(classifierJson);
			Classifier classifierFromJson = serializer.readValue(classifierJson, Classifier.class);
			
			
			ovaLearner.learn(trainParts[1]);
			evaluator.clear();
			for (Example e : testSet.getExamples()) {
				// Predict the class
				ClassificationOutput p = f.predict(e);
				evaluator.addCount(e, p);
			}
			System.out.println("Accuracy after second part training: " + evaluator.getAccuracy());
			OneVsAllLearning ovaLearnerFromJson = serializer.readValue(jsonLearner, OneVsAllLearning.class);
			ovaLearnerFromJson.setPredictionFunction(classifierFromJson);
			MulticlassClassificationEvaluator evaluatorFromJson = new MulticlassClassificationEvaluator(
					trainingSet.getClassificationLabels());
			for (Example e : testSet.getExamples()) {
				// Predict the class
				ClassificationOutput p = classifierFromJson.predict(e);
				evaluatorFromJson.addCount(e, p);
			}
			System.out.println("Accuracy model from JSON after first part training: " + evaluatorFromJson.getAccuracy());
			ovaLearnerFromJson.learn(trainParts[1]);
			
			evaluatorFromJson.clear();
			for (Example e : testSet.getExamples()) {
				// Predict the class
				ClassificationOutput p = classifierFromJson.predict(e);
				evaluatorFromJson.addCount(e, p);
			}
			System.out.println("Accuracy model from JSON after second part training: " + evaluatorFromJson.getAccuracy());
			
			
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
