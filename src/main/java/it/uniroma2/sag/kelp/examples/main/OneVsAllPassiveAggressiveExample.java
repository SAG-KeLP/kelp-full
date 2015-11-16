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
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.standard.RbfKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive.KernelizedPassiveAggressiveClassification;
import it.uniroma2.sag.kelp.predictionfunction.classifier.multiclass.OneVsAllClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.multiclass.OneVsAllClassifier;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

import java.util.List;

/**
 * This example illustrates how to perform multiclass classification,
 * with a One-Vs-All strategy with the Passive Aggressive Algorithm.
 * 
 * @author Giuseppe Castellucci, Danilo Croce
 */
public class OneVsAllPassiveAggressiveExample {

	public static void main(String[] args) {
		try {
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/multiplerepresentation/train.klp");
			// Read a dataset into a test variable
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/multiplerepresentation/test.klp");

			List<Label> classes = trainingSet.getClassificationLabels();

			
			for (int i=0; i<classes.size(); ++i) {
				Label l = classes.get(i);
				System.out.println("Class: " + l.toString());
				System.out.println(trainingSet.getNumberOfPositiveExamples(l));
				System.out.println(testSet.getNumberOfPositiveExamples(l));
			}
			
			// instantiate a passive aggressive algorithm
			KernelizedPassiveAggressiveClassification kPA = new KernelizedPassiveAggressiveClassification();
			// set an aggressiveness parameter
			kPA.setC(2f);

			// Kernel for the first representation (0-index)
			Kernel linear = new LinearKernel("0");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel = new NormalizationKernel(
					linear);
			// Apply a 2-degree Polynomial kernel on the score (normalized) computed by
			// the linear kernel
			Kernel polyKernel = new PolynomialKernel(2f, normalizedKernel);

			// Kernel for the second representation (1-index)
			Kernel linear1 = new LinearKernel("1");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel1 = new NormalizationKernel(
					linear1);
			// Apply a RBF kernel on the score (normalized) computed by
			// the linear kernel
			Kernel rbfKernel = new RbfKernel(2f, normalizedKernel1);
			// tell the algorithm that the kernel we want to use in learning is
			// the polynomial kernel

			LinearKernelCombination linearCombination = new LinearKernelCombination();
			linearCombination.addKernel(1f, polyKernel);
			linearCombination.addKernel(1f, rbfKernel);
			// normalize the weights such that their sum is 1
			linearCombination.normalizeWeights();
			
			// set the kernel for the PA algorithm
			kPA.setKernel(linearCombination);
			
			// Instantiate a OneVsAll learning algorithm
			// It is a so called meta learner, it receives in input a binary learning algorithm
			OneVsAllLearning metaOneVsAllLearner = new OneVsAllLearning();
			metaOneVsAllLearner.setBaseAlgorithm(kPA);
			metaOneVsAllLearner.setLabels(classes);

			long startLearningTime = System.currentTimeMillis();
			// learn and get the prediction function
			metaOneVsAllLearner.learn(trainingSet);
			OneVsAllClassifier f = metaOneVsAllLearner.getPredictionFunction();
			long endLearningTime = System.currentTimeMillis();

			// classify examples and compute some statistics
			MulticlassClassificationEvaluator ev = new MulticlassClassificationEvaluator(classes);
			for (Example e : testSet.getExamples()) {
				OneVsAllClassificationOutput prediction = f.predict(e);
				ev.addCount(e, prediction);
			}

			System.out
					.println("Accuracy: "
							+ ev.getAccuracy());
			System.out.println("Learning time without cache: " + (endLearningTime-startLearningTime) + " ms");
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
