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

import weka.core.Instance;
import weka.core.Instances;
import static java.lang.System.out;  
public class XgboostCV {
	
	static String arff=".arff";
	static String txt=".txt";
	static String csv=".csv";
	static String libsvm=".libsvm.txt";
	
	public static final int BUFSIZE = 1024 * 8; 
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		//参数
		int nFlod = 10;
		//原始整个训练集数据
//		String inputFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\";
//		String filename = "train_final_nouid";
		//C:\\Users\\dell\\Desktop\\competeDianzi\\data\\0201\\
		String inputFilePath ="";
		String filename = "train_final_0201";
//		//训练集添加样本
		String intxtfilePath = "";
		String addFilename ="unlable_final_0201_lable_0201_test";
		
		String outResultFile = "resultjilu.txt";

		Base_fuction.Out_file(outResultFile, "", false);
		
//		//将csv 转为arff
//	   Txt2Arff.txt2Arff(inputFilePath+filename+".csv", inputFilePath+filename+".arff");
//		
//		//将arff文件分折
//		Base_fuction source = new Base_fuction(filename,inputFilePath);		
//		source.partion_CV(0, nFlod, source.GetOri_instances(), filename, inputFilePath+"cv\\");
		
		
		//对每折进行训练预测
		float [] aucScore = new float[nFlod];
		for(int i = 0; i < nFlod;i++){
			
			System.out.println("Begin "+i+" cv");
			//将.arff 转为libsvm格式			
			String trainfile=filename+"_CV"+i+"_train";
			String testfile=filename+"_CV"+i+"_test";
//			Txt2LibSvm.uci2Libsvm(inputFilePath+"cv\\"+trainfile+arff,inputFilePath+"cv\\"+trainfile+libsvm);
//			Txt2LibSvm.uci2Libsvm(inputFilePath+"cv\\"+testfile+arff,inputFilePath+"cv\\"+testfile+libsvm);
			//添加新的训练样本
//			addSample2Train(intxtfilePath,inputFilePath+"cv\\"+trainfile+libsvm,addFilename);			
//			DMatrix trainMat = new DMatrix(inputFilePath+"cv\\"+trainfile+libsvm+".Merge");
			
			addSample2Train(intxtfilePath,inputFilePath+trainfile+libsvm,addFilename);			
			DMatrix trainMat = new DMatrix(inputFilePath+trainfile+libsvm+".Merge");
			
			
			
	        //load valid mat (svmlight format)
//			DMatrix trainMat = new DMatrix(inputFilePath+"cv\\"+trainfile+libsvm);
//	        DMatrix testMat = new DMatrix(inputFilePath+"cv\\"+testfile+libsvm);
			DMatrix testMat = new DMatrix(inputFilePath+testfile+libsvm);
//	        double rowRatio[]={0.5,1.0};
//	        double columnRatio[]={0.5,1.0};
	        int classifierNum=1;
	        float trainPredicts[][][]=new float[classifierNum][(int) trainMat.rowNum()][1];
	        float testPredicts[][][]=new float[classifierNum][(int) testMat.rowNum()][1];
	        
	        for(int j=0;j<classifierNum;j++){
	        	//set params
//	        	final double r=rowRatio[j/2];
//            	final double c=columnRatio[j%2];
            	
	        	Params param = new Params() {
		            {
		            	put("booster", "gbtree");
						put("eta", 0.02);
//						put("scale_pos_weight", 8.7);//8.7
						put("lambda", 700);
						put("subsample", 0.7);
						put("colsample_bytree", 0.30);
						put("min_child_weight", 1);// 5
						put("max_depth", 8);
						put("silent", 1);
						put("nthread", 40);
						put("objective", "binary:logistic");
						put("gamma", 0);
						put("eval_metric", "auc");
						// put("max_delta_step", 1);//0 1-10
		            }
		        };
		        
		        //set round
		        int round = 1520;	        
		        //specify watchList
		        List<Map.Entry<String, DMatrix>> watchs =  new ArrayList<>();
//		        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMat));
		        watchs.add(new AbstractMap.SimpleEntry<>("test", testMat));  
		        
		         
		        Booster booster = Trainer.train(param, trainMat, round, watchs, null, null);
		      //predict
		        trainPredicts[j] = booster.predict(trainMat);  
		        testPredicts[j] = booster.predict(testMat);  
		        
				float[][] testPredictsNew = new float[(int) testMat.rowNum()][1];

				for (int k = 0; k < (int) (int) testMat.rowNum(); k++) {
					testPredictsNew[k][0] =  testPredicts[j][k][0];
				}
		        
		        AUCEval eval = new AUCEval();	
		        aucScore[i] = eval.evalAUC(testPredictsNew, testMat.getLabel());
		        System.out.println("!!!!!cv\t"+i+"\tauc=" + aucScore[i]+"\n");
		        Base_fuction.Out_file(outResultFile, "!!!!!cv\t"+i+"\tauc=" + aucScore[i]+"\n", true);
	        }
	        
//	        double baseClassiferWeight[]=PSO_feature_weight.excute(classifierNum, testPredicts,testMat.getLabel());
//	        float testEnsemblePredicts[][]=Base_fuction.caculate_ensemble_result(testPredicts, baseClassiferWeight);
//	        AUCEval eval = new AUCEval();	
//	        aucScore[i] = eval.eval(testEnsemblePredicts, testMat);
//	        System.out.println("Fianl auc=" + aucScore[i]+"\n");   
			
		}
		
		System.out.println("aucAvg="+calAvgAuc(aucScore)+"\n");
		Base_fuction.Out_file(outResultFile, "aucAvg="+calAvgAuc(aucScore)+"\n", true);
		
	}
	
	public static float calAvgAuc(float [] aucScore){
		
		float sum = 0f;
		for(int i = 0;i < aucScore.length; i++){
			sum+= aucScore[i];
		}		
		return sum/aucScore.length;
	}
	
	public static void addSample2Train(String intxtfilePath,String inlibsvmfilePath,String addFilename) throws IOException{
		
		Txt2LibSvm.uci2Libsvm(intxtfilePath+addFilename+csv,intxtfilePath+addFilename+libsvm);		
		mergeFiles(inlibsvmfilePath+".Merge",new String[]{intxtfilePath+addFilename+libsvm,inlibsvmfilePath});		
		
	}
	
	
     
	    public static void mergeFiles(String outFile, String[] files) {  
	        FileChannel outChannel = null;  
	        out.println("Merge " + Arrays.toString(files) + " into " + outFile);  
	        try {  
	            outChannel = new FileOutputStream(outFile).getChannel();  
	            for(String f : files){  
	                FileChannel fc = new FileInputStream(f).getChannel();   
	                ByteBuffer bb = ByteBuffer.allocate(BUFSIZE);  
	                while(fc.read(bb) != -1){  
	                    bb.flip();  
	                    outChannel.write(bb);  
	                    bb.clear();  
	                }  
	                fc.close();  
	            }  
	            out.println("Merged!! ");  
	        } catch (IOException ioe) {  
	            ioe.printStackTrace();  
	        } finally {  
	            try {if (outChannel != null) {outChannel.close();}} catch (IOException ignore) {}  
	        }  
	    }  
	

}
