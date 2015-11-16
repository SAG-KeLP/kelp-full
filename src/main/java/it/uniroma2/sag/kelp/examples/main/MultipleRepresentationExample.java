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

package it.uniroma2.sag.kelp.examples.main;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.standard.RbfKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive.KernelizedPassiveAggressiveClassification;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;

/**
 * 
 * KeLP supports natively a multiple representation formalism. 
 * It is useful, for example, when the same data can be represented by different 
 * observable properties. 
 * <p>
 * For example, in NLP one can decide to derive features of a sentence for 
 * different syntactic levels (e.g. part-of-speech, chunk, dependency) and 
 * treat them in a learning algorithms with different kernel functions.
 * 
 * This example illustrates how to leverage on KeLP ability to work
 * with multiple representations.
 * 
 * @author Giuseppe Castellucci, Danilo Croce
 */
public class MultipleRepresentationExample {

	public static void main(String[] args) {
		try {
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/multiplerepresentation/train.klp");
			// Read a dataset into a test variable
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/multiplerepresentation/test.klp");

			// define the positive class
			StringLabel positiveClass = new StringLabel("food");

			// print some statistics
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());
			System.out.print("Positive examples ");
			System.out.println(trainingSet
					.getNumberOfPositiveExamples(positiveClass));
			System.out.print("Negative examples ");
			System.out.println(trainingSet
					.getNumberOfNegativeExamples(positiveClass));

			System.out.println("Test set statistics");
			System.out.print("Examples number ");
			System.out.println(testSet.getNumberOfExamples());
			System.out.print("Positive examples ");
			System.out.println(testSet
					.getNumberOfPositiveExamples(positiveClass));
			System.out.print("Negative examples ");
			System.out.println(testSet
					.getNumberOfNegativeExamples(positiveClass));

			// instantiate a passive aggressive algorithm
			KernelizedPassiveAggressiveClassification kPA = new KernelizedPassiveAggressiveClassification();
			// indicate to the learner what is the positive class
			kPA.setLabel(positiveClass);
			// set an aggressiveness parameter
			kPA.setC(2f);

			// Kernel for the first representation (0-index)
			Kernel linear = new LinearKernel("0");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel = new NormalizationKernel(
					linear);
			// Apply a Polynomial kernel on the score (normalized) computed by
			// the linear kernel
			Kernel polyKernel = new PolynomialKernel(2f, normalizedKernel);

			// Kernel for the second representation (1-index)
			Kernel linear1 = new LinearKernel("1");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel1 = new NormalizationKernel(
					linear1);
			// Apply a Polynomial kernel on the score (normalized) computed by
			// the linear kernel
			Kernel rbfKernel = new RbfKernel(1f, normalizedKernel1);
			// tell the algorithm that the kernel we want to use in learning is
			// the polynomial kernel

			LinearKernelCombination linearCombination = new LinearKernelCombination();
			linearCombination.addKernel(1f, polyKernel);
			linearCombination.addKernel(1f, rbfKernel);
			// normalize the weights such that their sum is 1
			linearCombination.normalizeWeights();
			
			// set the kernel for the PA algorithm
			kPA.setKernel(linearCombination);

			// learn and get the prediction function
			kPA.learn(trainingSet);
			Classifier f = kPA.getPredictionFunction();

			// classify examples and compute some statistics
			BinaryClassificationEvaluator ev = new BinaryClassificationEvaluator(positiveClass);
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
				ev.addCount(e, p);
			}

			System.out
					.println("Accuracy: "
							+ ev.getAccuracy());
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
