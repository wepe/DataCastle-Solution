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

import edu.cqupt.xdata.model.feature.FeatureSelected;
import weka.core.Instance;
import weka.core.Instances;
import static java.lang.System.out;  
public class XGBoostRandomTest {
	
	static String arff=".arff";
	static String txt=".txt";
	static String csv=".csv";
	static String libsvm=".libsvm.txt";
	
	public static final int BUFSIZE = 1024 * 8; 
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		
		

		
	
		//原始整个训练集数据
		String inputTrainFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\";
//		String inputTrainFilePath ="";

		String filename = "train_final_nouid";
		String inputTestFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\";
//		String inputTestFilePath ="";

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
			
			//将训练集随机选择特征
			int classifierNum=50;

			int select_feature_num=300;
			int g_feature_index[][]=new int[classifierNum][select_feature_num+1];
			
			for(int num=0;num<classifierNum;num++){
				int g_r[]=FeatureSelected.random_g_int(traincvInstances.numAttributes()-2,0,select_feature_num);	
				for(int j=0;j<select_feature_num;j++){
					g_feature_index[num][j]=g_r[j];
					System.out.print(g_feature_index[num][j]+"\t");
				}
				System.out.println();
				g_feature_index[num][select_feature_num]=traincvInstances.numAttributes()-1;  //类别标签
			}
			
			
			Instances gTrainAllIns[]=FeatureSelected.partion_subspace_ins(traincvInstances,g_feature_index);
			Instances gTrainIns[]=FeatureSelected.partion_subspace_ins(traincvsubInstances,g_feature_index);
			Instances gValidIns[]=FeatureSelected.partion_subspace_ins(validInstances,g_feature_index);
			Instances gTestIns[]=FeatureSelected.partion_subspace_ins(testcvInstances,g_feature_index);
			
			
	        //load valid mat (svmlight format)
//			DMatrix trainMat = new DMatrix(inputFilePath+"cv\\"+trainfile+libsvm);
//	        DMatrix testMat = new DMatrix(inputFilePath+"cv\\"+testfile+libsvm);
	        

	        
	        float validPredicts[][][]=new float[classifierNum][(int) validInstances.numInstances()][1];
	        float testPredicts[][][]=new float[classifierNum][(int) testcvInstances.numInstances()][1];
	        
	        
	        
	        float[] validLabel=new float [(int) validInstances.numInstances()];
	        float[] testLabel=new float [(int) testcvInstances.numInstances()];
	        
	       
	        
	        for(int j=0;j<classifierNum;j++){
	        	//set params
//	        	final double r=rowRatio[j/2];
//            	final double c=columnRatio[j%2];
	        	String trainallfileSubSpace=filename+"_CV_train_all"+j;	
				String trainfileSubSpace=filename+"_CV_train"+j;	
				String validfileSubSpace=filename+"_CV_valid"+j;	
				String testfileSubSpace=filename+"_CV_test"+j;
				
//        		Base_fuction.Outfile_instances_arff(gTrainAllIns[j], trainallfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
	        	Base_fuction.Outfile_instances_arff(gTrainIns[j], trainfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
		        Base_fuction.Outfile_instances_arff(gValidIns[j], validfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
		        Base_fuction.Outfile_instances_arff(gTestIns[j], testfileSubSpace, inputTrainFilePath+"SubspaceTest\\");
		        
//		        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+trainallfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+trainallfileSubSpace+libsvm);
		        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+trainfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+trainfileSubSpace+libsvm);
		        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+validfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+validfileSubSpace+libsvm);
		        Txt2LibSvm.uci2Libsvm(inputTrainFilePath+"SubspaceTest\\"+testfileSubSpace+arff,inputTrainFilePath+"SubspaceTest\\"+testfileSubSpace+libsvm);
	        	
				
//				DMatrix trainallMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+trainallfileSubSpace+libsvm);
				DMatrix trainMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+trainfileSubSpace+libsvm);
				DMatrix validMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+validfileSubSpace+libsvm);
		        DMatrix testMat = new DMatrix(inputTrainFilePath+"SubspaceTest\\"+testfileSubSpace+libsvm);
		        

	        	
	        	if(j==0){
	        		validLabel=validMat.getLabel();
	        		testLabel=testMat.getLabel();
	        	}
	        	 final int randmax_delta_step = Base_fuction.random_int(0, 10);
		        
	        	Params param = new Params() {
		            {
		            	put("booster","gbtree");
		                put("eta", 0.01);
		                put("scale_pos_weight", 8.7);
		                put("lambda", 700);
		                put("subsample", 0.7);
		                put("colsample_bytree", 0.70);
		                put("min_child_weight", 5);//5
		                put("max_depth", 8);
		                put("silent", 1);
		                put("nthread", 40);
		                put("objective", "binary:logistic");
		                put("gamma", 0);
		                put("eval_metric", "auc");
//		                put("max_delta_step", randmax_delta_step);//0 1-10
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
	     
//	        String outResultFile="C:\\Users\\dell\\Desktop\\competeDianzi\\20160128Result\\res0129_2.csv";
	        String outResultFile="res0129_3.csv";

			Base_fuction.Out_file(outResultFile, "", false);
			String res = "";
			for(int i = 0; i < testEnsemblePredicts.length;i++){
				res+= testEnsemblePredicts[i][0];
				res+= "\n";
			}
			Base_fuction.Out_file(outResultFile, res, true);    
				
		
	}
	


	
	

	

}
