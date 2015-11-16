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
package it.uniroma2.sag.kelp.main;

import java.io.File;
import java.io.PrintWriter;
import java.util.List;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.BinaryClassificationEvaluator;
import it.uniroma2.sag.kelp.utils.evaluation.Evaluator;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

public class Classify {
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.out.println("USAGE: datasetPath modelPath predictionsPath");
			System.exit(1);
		}

		// Initialize some parameters
		String testPath = args[0];
		String modelPath = args[1];
		String predictionsPath = args[2];

		// Load a dataset
		SimpleDataset testSet = new SimpleDataset();
		testSet.populate(testPath);

		// Instantiate a learning algorithm through a Json file
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		Classifier classifier = serializer.readValue(new File(modelPath), Classifier.class);

		List<Label> labels = classifier.getLabels();
		Evaluator ev = null;
		if (labels.size() > 1) {
			ev = new MulticlassClassificationEvaluator(labels);
		} else {
			ev = new BinaryClassificationEvaluator(labels.get(0));
		}
		PrintWriter pw = new PrintWriter(predictionsPath, "utf8");
		StringBuilder b = new StringBuilder();
		for (Example e : testSet.getExamples()) {
			b.delete(0, b.length());
			ClassificationOutput predict = classifier.predict(e);
			ev.addCount(e, predict);
			for (Label l : labels) {
				b.append(l + ":" + predict.getScore(l) + "\t");
			}
			
			pw.println(b.toString().substring(0, b.length()-1));
		}

		pw.flush();
		pw.close();
		
		System.out.println("Accuracy on test set: " + ev.getPerformanceMeasure("Accuracy"));
	}

}
