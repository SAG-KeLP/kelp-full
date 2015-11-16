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
import it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive.LinearPassiveAggressiveClassification;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;

/**
 * This is a very simple classification example based on a linear version of the
 * Passive Aggressive algorithm.
 * <p>
 * Dataset used are the ones used in the svmlight website. They have been
 * modified to be read by KeLP. In fact, a single row in KeLP must indicate what
 * kind of vectors your are using, Sparse or Dense. In the svmlight dataset
 * there are Sparse vectors, so if you open the train.klp and test.klp files you
 * can notice that each vector is enclosed in BeginVector (|BV|) and EndVector
 * (|EV|) tags.
 * 
 * @author Giuseppe Castellucci, Danilo Croce
 */
public class HelloLearning {

	public static void main(String[] args) {
		try {
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/hellolearning/train.klp");
			// Read a dataset into a test variable
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/hellolearning/test.klp");

			// define the positive class
			StringLabel positiveClass = new StringLabel("+1");

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
			LinearPassiveAggressiveClassification passiveAggressiveAlgorithm = new LinearPassiveAggressiveClassification();
			// use the first (and only here) representation
			passiveAggressiveAlgorithm.setRepresentation("0");
			// indicate to the learner what is the positive class
			passiveAggressiveAlgorithm.setLabel(positiveClass);
			// set an aggressiveness parameter
			passiveAggressiveAlgorithm.setC(0.01f);

			// learn and get the prediction function
			passiveAggressiveAlgorithm.learn(trainingSet);
			Classifier f = passiveAggressiveAlgorithm.getPredictionFunction();
			// classify examples and compute some statistics
			int correct = 0;
			BinaryClassificationEvaluator ev = new BinaryClassificationEvaluator(positiveClass);
			int predicted = 0;
			int tobe = 0;
			int cf1=0;
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
				if (e.isExampleOf(positiveClass))
					tobe++;
				if (p.getScore(positiveClass) >= 0)
					predicted++;
				if (p.getScore(positiveClass) >= 0
						&& e.isExampleOf(positiveClass)) {
					correct++;
					cf1++;
				} else if (p.getScore(positiveClass) < 0
						&& !e.isExampleOf(positiveClass))
					correct++;

				ev.addCount(e, p);
			}
			System.out
					.println("Accuracy: "
							+ ((float) correct / (float) testSet
									.getNumberOfExamples()));

			float prec = (float)cf1/(float)predicted;
			float rec = (float)cf1/(float)tobe;
			float f1 = 2*prec*rec/(prec+rec);
			
			System.out.println("ClassificationEvaluator Accuracy: "
					+ ev.getPerformanceMeasure("Accuracy"));
			System.out.println("InClass F1: "
					+ f1);
			System.out.println("ClassificationEvaluatorF1: "
					+ ev.getPerformanceMeasure("F1"));
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
