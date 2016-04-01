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
import zr.until.FileSpilt;
import zr.until.simfilter;
import static java.lang.System.out;

public class XgboostSimTest {

	static String arff = ".arff";
	static String txt = ".txt";
	static String csv = ".csv";
	static String libsvm = ".libsvm.txt";

	public static final int BUFSIZE = 1024 * 8;

	public static void main(String[] args) throws IOException, XGBoostError {

		// 参数
		int nNumFiles = 10;
		double oneThrod = 0.51;
		double zroeThrod = 0.99;

		// 原始整个训练集预测集数据
		String inputFilePath = "C:\\Users\\dell\\Desktop\\competeDianzi\\data\\0202\\";
		// String inputFilePath ="";
		String filenametrain = "train_final_nouid";
		String filenametest = "test_final_nouid";
		
		
		//将预测集进行拆分
//		FileSpilt.fileSpilt(inputFilePath, filenametest, inputFilePath+"spilt\\", nNumFiles);	
		
		
		

		String outResultFile =inputFilePath+ "\\result\\process_search_unlable_10_0.99_0.51.csv";

		Base_fuction.Out_file(outResultFile, "", false);


        
		// 将每折的训练集测试集读入内存
		DMatrix trainMatALL[] = new DMatrix[nNumFiles];
		DMatrix testMatALL[] = new DMatrix[nNumFiles];
		for (int i = 0; i < nNumFiles; i++) {
			System.out.println("Begin read data" + i + " cv");
			// 将.arff 转为libsvm格式
//			Txt2LibSvm.uci2Libsvm(
//					inputFilePath  + filenametrain + csv,
//					inputFilePath  + filenametrain + libsvm);
			
			Txt2LibSvm.uci2Libsvm(inputFilePath+"spilt\\"  + filenametest+"_"+i + csv,
					inputFilePath+"spilt\\"  + filenametest +"_"+i+ libsvm);
			
			//过滤新的训练集
			simfilter.simFilterProcess(inputFilePath+"spilt\\" + filenametest+"_"+i+"_esle"+ csv, inputFilePath+"spilt\\"  + filenametest+"_"+i+"_esle_filter"+ csv, oneThrod,zroeThrod);
			
//			
//			//添加新的训练样本
			addSample2Train(inputFilePath+"spilt\\",inputFilePath+filenametrain+libsvm,filenametest+"_"+i+"_esle_filter");			
			
			
			// load valid mat (svmlight format)
//			trainMatALL[i] = new DMatrix(inputFilePath  + filenametrain
//					+ libsvm);
			//读入新的训练集
		    trainMatALL[i] = new DMatrix(inputFilePath+filenametrain+libsvm+".Merge");
			
			testMatALL[i] = new DMatrix(inputFilePath+"spilt\\"  + filenametest +"_"+i+ libsvm);
			
			
		}
		System.out.println("Read data finished");

		float maxAucSore = 0f;

		Params param = new Params() {
			{
				put("booster", "gbtree");
				put("eta", 0.02);
				put("scale_pos_weight", 6);//8.7
				put("lambda", 700);
				put("subsample", 0.7);
				put("colsample_bytree", 0.30);
				put("min_child_weight", 1);// 5
				put("max_depth", 8);
				put("silent", 1);
				put("nthread", 20);
				put("objective", "binary:logistic");
				put("gamma", 0);
				put("eval_metric", "auc");
				// put("max_delta_step", 1);//0 1-10
			}
		};

		int round = 1520;

		for (int cvNum = 0; cvNum < nNumFiles; cvNum++) {
			// specify watchList
			List<Map.Entry<String, DMatrix>> watchs = new ArrayList<>();
			 watchs.add(new AbstractMap.SimpleEntry<>("train",
			 trainMatALL[cvNum]));
			// watchs.add(new AbstractMap.SimpleEntry<>("test",
			// testMatALL[cvNum]));

			Booster booster = Trainer.train(param, trainMatALL[cvNum], round,
					watchs, null, null);
			// predict
			float testPredicts[][] = booster.predict(testMatALL[cvNum]);

			float[][] testPredictsNew = new float[testPredicts.length][1];

			for (int j = 0; j < (int) testPredicts.length; j++) {
				testPredictsNew[j][0] = 1 - testPredicts[j][0];
			}
			String res = "\"score\"\n";
			if(cvNum !=0){
				res ="";			
				}
			
			for(int i = 0; i < testPredictsNew.length;i++){
				res+= testPredictsNew[i][0];
				res+= "\n";
			}
			Base_fuction.Out_file(outResultFile, res, true);   

		}

	}

	public static float calAvgAuc(float[] aucScore) {

		float sum = 0f;
		for (int i = 0; i < aucScore.length; i++) {
			sum += aucScore[i];
		}
		return sum / aucScore.length;
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

