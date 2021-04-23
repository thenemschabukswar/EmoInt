1. To install weka:
$ sudo apt-get install -y weka

2. To run:
$ cd weka-3-8-5
$ ./weka.sh

3. Download and install packages AffectiveTweets, LibLINEAR, LibSVM packages in Weka ->Tools ->Package Manager -> (select) ->Install 

4. Now to convert the officially given testing and training files for txt to wweka acceptable arff, follow this steps:
4.1 Edit the downloaded files (e.g. anger_train.txt) by adding for columns at the top: id	tweet	emotion	score
4.2 Edit the file names in the python script txt2csv.py , such as "anger_train.txt" and "anger_train.csv" for every 4 training and 4 testing files. Kindly note that files with intensity scores should be used only
4.3 Now run the python file for every 8 files
$ python3 txt2csv.py
4.4 Once the csv files are created, edit the extension from csv to arff, and remove the first row of the file. Now, as per the arff file format, first add @relation at top row and name the file accordingly (e.g. anger_train). Then add @attribute <attribute_name> <type> for all the attributes. And finally add @data after which as the rows of comma separated data can be placed. So, for example, the arff file content should be like:
@relation anger_train
@attribute id string
@attribute tweet string
@attribute emotion string
@attribute score numeric
@data
1000	I hate winter.	anger	0.340

5. Open weka as stated in step 2, then weka->explorer, click open files and open the <emotion>_training_1.arff files (or any other files). Now select tweet attribute, click on filter and choose MultiFilter. Now after clicking MultiFilter, click on filters, choose filters->unsupervised->TweetToLexiconFeatureVector and filters->unsupervised->TweetToSentiStrengthFeatureVector and click Add. Now press Apply. This will tokenize and generate embeddings of tweet into multiple attibutes. Now the attributes tweet, id and emotion can be removed as these are not required. Now this file can be saved in arff format using Save button. Such procedure need to followed for all the training and testing files. We followed the convention to add token in the file name (e.g. fear_test_token.arff). [video reference: https://www.youtube.com/watch?v=QtNYArb0Tkc]

6. Now open any <emotion>_train.arff file, then click Classify -> Supplied test set -> Set -> Open file. Now upload respective <emotion>_test.arff file and select score class. Choose the classifier as trees->RandomForest or functions ->LibLINEAR ->Regression (primal). Then hit close. The select score class over the start button and enter Start button. This will execute and show the test results. This results in baseline feature results.

7. The output of GRU/CNN-GRU models are added to the baseline feature files in the following manner:

7.1 First the baseline feature arff files need to be converted to txt files by changing the extension from arff to txt. Then all the arff file specific contents (i.e. @relation, @attribute, @data) should be removed (preferably stored in a different file to use later) so that only data remains in such a a format:
{0 <numeric_value>,1 <<numeric_value>, ...... ,45 <numeric_value>}

7.2 Now, for every such files, run the following command:
$ sed "s/}/,/g" <emotion>_test_token.txt > <emotion>_test_token_2.txt

7.3 The output of GRU/ CNN-GRU files should be converted to csv files as stated in steps 4.1 to 4.3. Then the top row of csv files were removed and saved.

7.4 Now, the csv files should be edited as:
$ sed 's/^/46 /; s/,/,47 /; s/,/,48 /2; s/,/,49 /3; s/$/}/' <emotion>__test_gru.csv > <emotion>__test_gru_back.txt

7.5 To ultimately paste the attributes at the end:
$ paste <emotion>_test_token_2.txt <emotion>_test_gru_back.txt > <emotion>_test_gru.txt

7.6 By repeating the steps 4.1 to 4.4, these txt files are again converted to arff files. Here, add the arff specific contents which were removed in step 7.1 and add the following attributes below the listed attribute and above the @data:
@attribute anger_gru numeric
@attribute fear_gru numeric
@attribute joy_gru numeric
@attribute sadness_gru numeric


8. Now as per step 6, run these files accordingly to test the files with extended attributes.
