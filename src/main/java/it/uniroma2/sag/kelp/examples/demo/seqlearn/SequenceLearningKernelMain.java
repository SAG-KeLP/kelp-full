/*
 * Copyright 2016 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.examples.demo.seqlearn;

import java.io.File;

import it.uniroma2.sag.kelp.data.dataset.SequenceDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SequenceExample;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.SimpleDynamicKernelCache;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.hmm.SequenceClassificationKernelBasedLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.predictionfunction.SequencePrediction;
import it.uniroma2.sag.kelp.predictionfunction.SequencePredictionFunction;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassSequenceClassificationEvaluator;

/**
 * 
 * This class shows how to use a
 * <code>SequenceClassificationLearningAlgorithm</code>. Given a dataset of
 * <code>SequenceExample</code>s where each item in the sequence is represented
 * as a feature vector, <b>the following code implements a kernel based learning
 * algorithm </b>. <br>
 * During the training step, each example is enriched with an additional
 * representation to consider the example history, in terms of classes assigned
 * to the previous example(s). The overall kernel function is thus a combination
 * of a kernel operating on the original representation and a kernel operating
 * on this additional representation. <br>
 * During the tagging process, the history of an example is dynamically
 * estimated by a classifier and the entire sequence of labels is derived
 * through a Viterbi Decoding step combined with a Beam Search Strategy.
 * 
 * <br>
 * <br>
 * The datasets used in this example have been created starting from the dataset
 * produced by Thorsten Joachims as an example problem for his SVM^{hmm}
 * implementation.<br>
 * 
 * The original dataset can be downloaded at:<br>
 * http://download.joachims.org/svm_hmm/examples/example7.tar.gz <br>
 * while its description is reported at:<br>
 * https://www.cs.cornell.edu/people/tj/svm_light/svm_hmm.html
 * 
 * @author Danilo Croce
 */
public class SequenceLearningKernelMain {

	public static void main(String[] args) throws Exception {

		String inputTrainFilePath = "src/main/resources/sequence_learning/declaration_of_independence.klp";
		String inputTestFilePath = "src/main/resources/sequence_learning/gettysburg_address.klp";
		String classificationOutputFile = "src/main/resources/sequence_learning/seq_labeling_kernel_based_classifier.klp";
		
		/*
		 * Given a targeted item in the sequence, this variable determines the
		 * number of previous example considered in the learning/labeling
		 * process.
		 * 
		 * NOTE: if this variable is set to 0, the learning process corresponds
		 * to a traditional multi-class classification schema.
		 */
		int transitionsOrder = 1;

		/*
		 * This variable determines the importance of the transition-based
		 * kernel during the learning process. Higher valuers will assign more
		 * importance to the transitions.
		 */
		float weight = 1f;

		/*
		 * The size of the beam to be used in the decoding process. This number
		 * determines the number of possible sequences produced in the labeling
		 * process. It will also increase the process complexity.
		 */
		int beamSize = 5;

		/*
		 * During the labeling process, each item is classified with respect to
		 * the target classes. To reduce the complexity of the labeling process,
		 * this variable determines the number of classes that received the
		 * highest classification scores to be considered after the
		 * classification step in the Viterbi Decoding.
		 */
		int maxEmissionCandidates = 3;

		/*
		 * This representation contains the feature vector representing items in
		 * the sequence
		 */
		String originalRepresentationName = "rep";

		SequenceDataset sequenceTrainDataset = new SequenceDataset();
		sequenceTrainDataset.populate(inputTrainFilePath);

		/*
		 * Instance classifier: Kernel based version. A kernel operating both on
		 * the original representation and on the artificial "history-based"
		 * representation is used. Here, such kernel is implemented as a linear
		 * combinations of the aforementioned kernels.
		 */
		Kernel representationKernel = new LinearKernel(originalRepresentationName);

		float cSVM = 1f;
		BinaryCSvmClassification binaryCSvmClassification = new BinaryCSvmClassification(representationKernel, null,
				cSVM, cSVM);

		/*
		 * Sequence classifier
		 */
		SequenceClassificationKernelBasedLearningAlgorithm sequenceClassificationLearningAlgorithm = new SequenceClassificationKernelBasedLearningAlgorithm(
				binaryCSvmClassification, transitionsOrder, weight);
		sequenceClassificationLearningAlgorithm.setBeamSize(beamSize);
		sequenceClassificationLearningAlgorithm.setMaxEmissionCandidates(maxEmissionCandidates);
		sequenceClassificationLearningAlgorithm.setKernelCache(new SimpleDynamicKernelCache());

		sequenceClassificationLearningAlgorithm.learn(sequenceTrainDataset);

		SequencePredictionFunction predictionFunction = (SequencePredictionFunction) sequenceClassificationLearningAlgorithm
				.getPredictionFunction();

		/*
		 * The classifier can be serialized. The following code shows how to
		 * serialize.
		 */
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		// how to save
		serializer.writeValueOnFile(predictionFunction, classificationOutputFile);
		// how to load
		predictionFunction = serializer.readValue(new File(classificationOutputFile), SequencePredictionFunction.class);

		/*
		 * Load the test set
		 */
		SequenceDataset sequenceTestDataset = new SequenceDataset();
		sequenceTestDataset.populate(inputTestFilePath);

		MulticlassSequenceClassificationEvaluator mcSeqEvaluator = new MulticlassSequenceClassificationEvaluator(
				sequenceTestDataset.getClassificationLabels());

		/*
		 * Tagging and evaluating
		 */
		for (Example example : sequenceTestDataset.getExamples()) {

			SequenceExample sequenceExample = (SequenceExample) example;
			SequencePrediction sequencePrediction = (SequencePrediction) predictionFunction.predict(sequenceExample);

			System.out.println(sequencePrediction);

			mcSeqEvaluator.addCount(sequenceExample, sequencePrediction);
		}

		System.out.println("Accuracy " + mcSeqEvaluator.getAccuracy());
	}

}
