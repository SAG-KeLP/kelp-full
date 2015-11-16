kelp-full
=========

 **KeLP** is the Kernel-based Learning Platform developed in the [Semantic Analytics Group][sag-site] of
the [University of Roma Tor Vergata][uniroma2-site]. 

This is a complete package of **KeLP**. 
It aggregates the following modules:

* [kelp-core](https://github.com/SAG-KeLP/kelp-core): it contains the core interfaces and classes for algorithms, kernels and representations. It contains also the base set of classifiers, regressors and clustering algorithms. It serves as the main module to develop new kernel functions or new algorithms.

* [kelp-additional-kernels](https://github.com/SAG-KeLP/kelp-additional-kernels): it contains additional kernel functions, such as the Tree Kernels or the Graph Kernels.

* [kelp-additional-algorithms](https://github.com/SAG-KeLP/kelp-additional-algorithms): it contains additional learning algorithms, such as the **KeLP** Java implementation of Liblinear or Online Learning algorithms, such as the Passive Aggressive.

##Working examples

This packages contains a set of fully functioning examples showing how to implement a learning system with KeLP. Batch learning algorithm as well as online learning algorithms usage is shown here. Different examples cover the usage of standard kernel, tree kernels and sequence kernel, with caching mechanisms.

Clone this project to obtain access to these examples with the complete datasets with:

```
git clone https://github.com/SAG-KeLP/kelp-full.git
```

NOTE: many of the provided examples require some memory in order to load the datasets and set up the kernel cache. You can assign memory to the Java Virtual Machine (JVM) using the option -Xmx. For instance -Xmx2G will provide 2G of memory to the JVM. In Eclipse such parameter shuld be written in Run->Run Configurations->Arguments->VM arguments.

#### Classification:
* **QuestionClassification** (it.uniroma2.sag.kelp.examples.demo.qc): this class implements the Question Classification demo. It includes both kernel operating on vectors and kernel operating on trees (stk and ptk).
* **QuestionClassificationLearningFromJson** (it.uniroma2.examples.demo.qc): the same demo as QuestionClassification with the difference that the learning algorithm specification is read from a Json file.
* **RCV1BinaryTextCategorizationLibLinear**, **RCV1BinaryTextCategorizationPA**,  **RCV1BinaryTextCategorizationPegasos** and **RCV1BinaryTextCategorizationDCD** (it.uniroma2.sag.kelp.examples.demo.rcv1) are examples of binary classifiers on the RCV1 dataset that can be found on the LibLinear website. These classes perform a N-Fold Cross Validation and show KeLP facilities to divide a dataset in N-Fold.
* **TweetSentimentAnalysisSemeval2013** (it.uniroma2.sag.kelp.examples.demo.tweetsent2013): a demo with multiple kernels and multiple classes on a dataset for the Twitter Sentiment Analysis task of Semeval2013.
* **OneVsAllSVMExample** (it.uniroma2.sag.kelp.examples.main): an example that shows the usage of the OneVsAll strategy with SVM over the IRIS dataset for a multiclassification schema.
* **SequenceKernelExample** (it.uniroma2.sag.kelp.examples.main): an example that shows the usage of a Sequence Kernel.
* **MultipleRepresentationExample** (it.uniroma2.sag.kelp.examples.main): a basic example showing the usage of multiple representations with multiple kernel functions with a PassiveAggressive algorithm.
* **KernelCacheExample** (it.uniroma2.sag.kelp.examples.main): an example that shows the usage of the KernelCache class to store the already computed kernel values between instances.
* **MutagClassification** 
(it.uniroma2.sag.kelp.examples.demo.mutag); an example that shows the application of graph kernels to the mutag dataset

#### Regression:
* **EpsilonSVRegressionExample** (it.uniroma2.sag.kelp.examples.demo.regression): This class contains an example of the usage of the Regression Example. The regressor implements the e-SVR learning algorithm discussed in [CC Chang & CJ Lin, 2011]. In this example a dataset is loaded from a file and then split in train and test.

#### Clustering:
* **KernelBasedClusteringExample** (it.uniroma2.sag.kelp.examples.demo.clustering): this class contains an example of the usage of the Kernel-based clustering.
* **LinearKMeansClusteringExample** (it.uniroma2.sag.kelp.examples.demo.clustering): this class contains an example of the usage of the Linear K-means clustering.


#### General Purpose:
* **ClassificationDemo** (it.uniroma2.sag.kelp.examples.main): it is a meta-learner that takes in input a Json description and a dataset.
* **Learn** (it.uniroma2.sag.kelp.main): the main file for learning a model. It takes in input a training dataset, a learning algorithm description in JSON and the path where the model will be saved.
* **Classify** (it.uniroma2.sag.kelp.main): the main file for classification. It takes in input the dataset to be classified, a previously learned model and the path where to store the final classifications.


##Including KeLP in your project

If you want to include the full functionalities of **KeLP** you can  easily include it in your [Maven][maven-site] project adding the following repositories to your pom file:

```
<repositories>
	<repository>
			<id>kelp_repo_snap</id>
			<name>KeLP Snapshots repository</name>
			<releases>
				<enabled>false</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>warn</checksumPolicy>
			</releases>
			<snapshots>
				<enabled>true</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>fail</checksumPolicy>
			</snapshots>
			<url>http://sag.art.uniroma2.it:8081/artifactory/kelp-snapshot/</url>
		</repository>
		<repository>
			<id>kelp_repo_release</id>
			<name>KeLP Stable repository</name>
			<releases>
				<enabled>true</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>warn</checksumPolicy>
			</releases>
			<snapshots>
				<enabled>false</enabled>
				<updatePolicy>always</updatePolicy>
				<checksumPolicy>fail</checksumPolicy>
			</snapshots>
			<url>http://sag.art.uniroma2.it:8081/artifactory/kelp-release/</url>
		</repository>
	</repositories>
```

Then, the [Maven][maven-site] dependency for the whole **KeLP** package:

```
<dependency>
    <groupId>it.uniroma2.sag.kelp</groupId>
    <artifactId>kelp-full</artifactId>
    <version>2.0.0</version>
</dependency>
```

Alternatively, thanks to the modularity of **KeLP**, you can include a fine grain selection of its modules adding to your POM files only the dependancies you need among the modules stated above.  

[sag-site]: http://sag.art.uniroma2.it "SAG site"
[uniroma2-site]: http://www.uniroma2.it "University of Roma Tor Vergata"
[maven-site]: http://maven.apache.org "Apache Maven"
