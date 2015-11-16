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
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.ObjectSerializer;

import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * This example illustrates how to serialize and deserialize learning algorithms, as well as
 * classification functions. The example is based on the OneVsAllSVMExample class.
 * 
 * @author Giuseppe Castellucci, Danilo Croce
 */
public class SerializationExample {

	public static void main(String[] args) {
		try {
			// Read a dataset into a trainingSet variable
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/iris_dataset/iris_train.klp");
			
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/iris_dataset/iris_test.klp");

			// print some statistics
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());
			
			List<Label> classes = trainingSet.getClassificationLabels();
			
			for (Label l : classes) {
				System.out.println("Training Label " + l.toString() + " " + trainingSet.getNumberOfPositiveExamples(l));
				System.out.println("Training Label " + l.toString() + " " + trainingSet.getNumberOfNegativeExamples(l));
				
				System.out.println("Test Label " + l.toString() + " " + testSet.getNumberOfPositiveExamples(l));
				System.out.println("Test Label " + l.toString() + " " + testSet.getNumberOfNegativeExamples(l));
			}
			
			// Kernel for the first representation (0-index)
			Kernel linear = new LinearKernel("0");
			// Normalize the linear kernel
			NormalizationKernel normalizedKernel = new NormalizationKernel(
					linear);
			// instantiate an svmsolver
			BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
			svmSolver.setKernel(normalizedKernel);
			svmSolver.setCp(2);
			svmSolver.setCn(1);
			
			OneVsAllLearning ovaLearner = new OneVsAllLearning();
			ovaLearner.setBaseAlgorithm(svmSolver);
			ovaLearner.setLabels(classes);
			
			// One can serialize the learning algorihtm
			ObjectSerializer serializer = new JacksonSerializerWrapper();
			String algoDescr = serializer.writeValueAsString(ovaLearner);
			System.out.println(algoDescr);
			
			// If the learning algorithm is specifified in the Json syntax
			// it is possible to load it in this way:
			ObjectMapper mapper = new ObjectMapper();
			ovaLearner = (OneVsAllLearning) mapper
					.readValue(algoDescr, OneVsAllLearning.class);
			// the ovaLearner object is the one loaded from the Json description
			// refer to the api documentation to read a learning algorithm from a file
			
			// learn and get the prediction function
			ovaLearner.learn(trainingSet);
			Classifier f = ovaLearner.getPredictionFunction();
			
			// it is possible also to serialize a classification function
			// that includes the model (e.g. the support vectors).
			String classificationFunctionDescr = serializer.writeValueAsString(f);
			System.out.println(classificationFunctionDescr);
			
			// and obiovously a classification function can be loaded in memory from
			// its json representation
			f = (Classifier)mapper.readValue(classificationFunctionDescr, Classifier.class);

			// classify examples and compute some statistics
			int correct = 0;
			for (Example e : testSet.getExamples()) {
				ClassificationOutput p = f.predict(testSet.getNextExample());
//				System.out.println(p.getPredictedClasses());
				if (e.isExampleOf(p.getPredictedClasses().get(0))) {
					correct++;
				}
			}

			System.out
					.println("Accuracy: "
							+ ((float) correct / (float) testSet
									.getNumberOfExamples()));
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}

}
