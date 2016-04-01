package edu.cqupt.xdata.model;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.IEvaluation;
import org.dmlc.xgboost4j.IObjective;
import org.dmlc.xgboost4j.demo.util.*;
import org.dmlc.xgboost4j.demo.CustomObjective.EvalError;
import org.dmlc.xgboost4j.demo.CustomObjective.LogRegObj;
import org.dmlc.xgboost4j.demo.util.AUCEval;
import org.dmlc.xgboost4j.demo.util.Base_fuction;
import org.dmlc.xgboost4j.demo.util.CustomEval;
import org.dmlc.xgboost4j.demo.util.Params;
import org.dmlc.xgboost4j.demo.util.Txt2Arff;
import org.dmlc.xgboost4j.demo.util.Txt2LibSvm;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.util.XGBoostError;

import Sampling.Sampling;
import edu.cqupt.xdata.model.feature.FeatureSelected;
import weka.core.Instance;
import weka.core.Instances;
import static java.lang.System.out;  
public class XGBoostSmoteTest {
	
	static String arff=".arff";
	static String txt=".txt";
	static String csv=".csv";
	static String libsvm=".libsvm.txt";
	
	public static final int BUFSIZE = 1024 * 8; 
	
	public static void main(String[] args) throws IOException, XGBoostError {	
	
		//原始整个训练集数据
		String inputTrainFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\";
		String filename = "train_final_nouid";
		String inputTestFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\";
		String testfilename = "test_final_nouid";		
			//将csv 转为arff
			Txt2Arff.txt2Arff(inputTrainFilePath+filename+".csv", inputTrainFilePath+filename+".arff");
			Txt2Arff.txt2Arff(inputTestFilePath+testfilename+".csv", inputTestFilePath+testfilename+".arff");		
			
			//将训练集测试集读入保存为instances
			Instances traincvInstances= new Base_fuction(filename,inputTrainFilePath).GetOri_instances();
			Instances testcvInstances= new Base_fuction(testfilename,inputTestFilePath).GetOri_instances();
				     
		
           
			//将训练集分为验证集与训练集
			int folds=10;
			Random rand=new Random(1);
			Instances rand_ins=new Instances(traincvInstances);
			rand_ins.randomize(rand);
			rand_ins.stratify(folds);
			
			Instances traincvsubInstances=rand_ins.trainCV(folds, 0);
			Instances validInstances=rand_ins.testCV(folds, 0);
			
			
			//Smote过采样

			
			//将训练集随机选择特征
			
			double g_smote_sample_ratio[]={1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0}; //somte采样比例

			int classifierNum=g_smote_sample_ratio.length;        

	        
	        float validPredicts[][][]=new float[classifierNum][(int) validInstances.numInstances()][1];
	        float testPredicts[][][]=new float[classifierNum][(int) testcvInstances.numInstances()][1];
	        
	        
	        
	        float[] validLabel=new float [(int) validInstances.numInstances()];
	        float[] testLabel=new float [(int) testcvInstances.numInstances()];
	        
	       
			String validfileSubSpace=filename+"_CV_valid";	
			String testfileSubSpace=filename+"_CV_test";
	        Base_fuction.Outfile_instances_arff(validInstances, validfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
	        Base_fuction.Outfile_instances_arff(testcvInstances, testfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
	        
	        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+validfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+validfileSubSpace+libsvm);
	        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+testfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+testfileSubSpace+libsvm);
	        
	        DMatrix validMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+validfileSubSpace+libsvm);
	        DMatrix testMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+testfileSubSpace+libsvm);
	        
	        for(int j=0;j<classifierNum;j++){

				System.out.println("Smote过采样");
				Instances smote_train_ins=Sampling.smote(traincvsubInstances, "1", g_smote_sample_ratio[j]);
				Base_fuction.print_instances_info(smote_train_ins);
	        	
	        	
	        	String trainallfileSubSpace=filename+"_CV_train_all"+j;	
				String trainfileSubSpace=filename+"_CV_train"+j;	

				
//        		Base_fuction.Outfile_instances_arff(gTrainAllIns[j], trainallfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
	        	Base_fuction.Outfile_instances_arff(smote_train_ins, trainfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
		       
//		        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+trainallfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+trainallfileSubSpace+libsvm);
		        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+trainfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+trainfileSubSpace+libsvm);
		        
	        	
				
//				DMatrix trainallMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+trainallfileSubSpace+libsvm);
				DMatrix trainMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+trainfileSubSpace+libsvm);
				        

	        	
	        	if(j==0){
	        		validLabel=validMat.getLabel();
	        		testLabel=testMat.getLabel();
	        	}
	        	 final int randmax_delta_step = Base_fuction.random_int(0, 10);
		        
	        	Params param = new Params() {
		            {
		            	put("booster","gbtree");
		                put("eta", 0.02);
		                put("scale_pos_weight", 8.7);
		                put("lambda", 700);
		                put("subsample", 0.7);
		                put("colsample_bytree", 0.30);
		                put("min_child_weight", 5);//5
		                put("max_depth", 8);
		                put("silent", 1);
		                put("nthread", 20);
		                put("objective", "binary:logistic");
		                put("gamma", 0);
		                put("eval_metric", "auc");
		                put("max_delta_step", randmax_delta_step);//0 1-10
		            }
		        };
		        
		        //set round
		        int round = 1600;	        
		        //specify watchList
		        List<Map.Entry<String, DMatrix>> watchs =  new ArrayList<>();
		        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMat));
		        watchs.add(new AbstractMap.SimpleEntry<>("valid", validMat)); 
//		        watchs.add(new AbstractMap.SimpleEntry<>("test", testMat));  
		        
		         
		        Booster booster = Trainer.train(param, trainMat, round, watchs, null, null);
//		        Booster boosterfianal = Trainer.train(param, trainallMat, round, watchs, null, null);

		      //predict
//		        trainPredicts[j] = booster.predict(trainMat);  
		        validPredicts[j] = booster.predict(validMat); 
		        testPredicts[j] = booster.predict(testMat);  
		        
		        AUCEval eval = new AUCEval();	
		        float validAuc = eval.evalAUC(validPredicts[j], validMat.getLabel());
		        System.out.println("!!BaseClassifer\t"+j+"\tauc=" + validAuc+"\n");
		        
	        }
	        
	        double baseClassiferWeight[]=PSO_feature_weight.excute(classifierNum, validPredicts,validLabel);
	       
	        float[][][] testPredictsNew = new float[classifierNum][(int) testcvInstances.numInstances()][1];
	        
	        for(int i = 0;i <classifierNum;i++){
	        	for(int j = 0;j < (int) testcvInstances.numInstances();j++){
	        		testPredictsNew[i][j][0]= 1-testPredicts[i][j][0];
	        	}
	        }
	        
	        float testEnsemblePredicts[][]=Base_fuction.caculate_ensemble_result(testPredictsNew, baseClassiferWeight);
	     
	        String outResultFile="C:\\Users\\dell\\Desktop\\competeDianzi\\20160128Result\\res0129_2.csv";
			Base_fuction.Out_file(outResultFile, "", false);
			String res = "";
			for(int i = 0; i < testEnsemblePredicts.length;i++){
				res+= testEnsemblePredicts[i][0];
				res+= "\n";
			}
			Base_fuction.Out_file(outResultFile, res, true);    
				
		
	}
	

	

}
