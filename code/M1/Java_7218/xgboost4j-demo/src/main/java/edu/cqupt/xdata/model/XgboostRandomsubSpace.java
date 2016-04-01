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
public class XgboostRandomsubSpace {
	
	static String arff=".arff";
	static String txt=".txt";
	static String csv=".csv";
	static String libsvm=".libsvm.txt";
	
	public static final int BUFSIZE = 1024 * 8; 
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		boolean IsSplitFile=true;
		
		String outResultFile="C:\\Users\\dell\\Desktop\\competeDianzi\\20160128Result\\res.txt";
		Base_fuction.Out_file(outResultFile, "", false);
		
		//参数
		int nFlod = 10;
		//原始整个训练集数据
		String inputFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\";
		String filename = "train_final_nouid";
		//训练集添加样本
/*		String intxtfilePath = "C:\\Users\\dell\\Desktop\\competeDianzi\\20160127\\";
		String addFilename ="70_train";*/
		
		if(IsSplitFile){
			//将csv 转为arff
			Txt2Arff.txt2Arff(inputFilePath+filename+".csv", inputFilePath+filename+".arff");
			
			//将arff文件分折
			Base_fuction source = new Base_fuction(filename,inputFilePath);		
			source.partion_CV(0, nFlod, source.GetOri_instances(), filename, inputFilePath+"cv\\");
		}
		
		

		
		//对每折进行训练预测
		float [] aucScore = new float[nFlod];
		for(int i = 0; i < nFlod;i++){
			
			Base_fuction.Out_file(outResultFile, "CV"+i+":\n", true);
			
			System.out.println("Begin "+i+" cv");
			//将.arff 转为libsvm格式			
			String trainfile=filename+"_CV"+i+"_train";			
			String testfile=filename+"_CV"+i+"_test";
			
//			Txt2LibSvm.uci2Libsvm(inputFilePath+"cv\\"+trainfile+arff,inputFilePath+"cv\\"+trainfile+libsvm);
//			Txt2LibSvm.uci2Libsvm(inputFilePath+"cv\\"+testfile+arff,inputFilePath+"cv\\"+testfile+libsvm);
	     
			//将训练集读入保存为instances
			Instances traincvInstances= new Base_fuction(trainfile,inputFilePath+"cv\\").GetOri_instances();
			Instances testcvInstances= new Base_fuction(testfile,inputFilePath+"cv\\").GetOri_instances();
           
			//将训练集分为验证集与训练集
			int folds=10;
			Random rand=new Random(1);
			Instances rand_ins=new Instances(traincvInstances);
			rand_ins.randomize(rand);
			rand_ins.stratify(folds);
			
			traincvInstances=rand_ins.trainCV(folds, 0);
			Instances validInstances=rand_ins.testCV(folds, 0);
			
			//将训练集随机选择特征
			int classifierNum=20;

			int select_feature_num=500;
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
			
			
			
			Instances gTrainIns[]=FeatureSelected.partion_subspace_ins(traincvInstances,g_feature_index);
			Instances gValidIns[]=FeatureSelected.partion_subspace_ins(validInstances,g_feature_index);
			Instances gTestIns[]=FeatureSelected.partion_subspace_ins(testcvInstances,g_feature_index);
			
			
	        //load valid mat (svmlight format)
//			DMatrix trainMat = new DMatrix(inputFilePath+"cv\\"+trainfile+libsvm);
//	        DMatrix testMat = new DMatrix(inputFilePath+"cv\\"+testfile+libsvm);
	        

	        
	        float trainPredicts[][][]=new float[classifierNum][(int) traincvInstances.numInstances()][1];
	        float validPredicts[][][]=new float[classifierNum][(int) validInstances.numInstances()][1];
	        float testPredicts[][][]=new float[classifierNum][(int) testcvInstances.numInstances()][1];
	        
	        
	        
	        float[] validLabel=new float [(int) validInstances.numInstances()];
	        float[] testLabel=new float [(int) testcvInstances.numInstances()];
	        
	        
	        
	        for(int j=0;j<classifierNum;j++){
	        	//set params
//	        	final double r=rowRatio[j/2];
//            	final double c=columnRatio[j%2];
				String trainfileSubSpace=filename+"_CV"+i+"_train"+j;	
				String validfileSubSpace=filename+"_CV"+i+"_valid"+j;	
				String testfileSubSpace=filename+"_CV"+i+"_test"+j;
				
	        	if(IsSplitFile){
	        		Base_fuction.Outfile_instances_arff(gTrainIns[j], trainfileSubSpace, inputFilePath+"Subspace\\");
		        	Base_fuction.Outfile_instances_arff(gValidIns[j], validfileSubSpace, inputFilePath+"Subspace\\");
		        	Base_fuction.Outfile_instances_arff(gTestIns[j], testfileSubSpace, inputFilePath+"Subspace\\");
		        	
		        	Txt2LibSvm.uci2Libsvm(inputFilePath+"Subspace\\"+trainfileSubSpace+arff,inputFilePath+"Subspace\\"+trainfileSubSpace+libsvm);
		        	Txt2LibSvm.uci2Libsvm(inputFilePath+"Subspace\\"+validfileSubSpace+arff,inputFilePath+"Subspace\\"+validfileSubSpace+libsvm);
		        	Txt2LibSvm.uci2Libsvm(inputFilePath+"Subspace\\"+testfileSubSpace+arff,inputFilePath+"Subspace\\"+testfileSubSpace+libsvm);
	        	}
				
 
				DMatrix trainMat = new DMatrix(inputFilePath+"Subspace\\"+trainfileSubSpace+libsvm);
				DMatrix validMat = new DMatrix(inputFilePath+"Subspace\\"+validfileSubSpace+libsvm);
		        DMatrix testMat = new DMatrix(inputFilePath+"Subspace\\"+testfileSubSpace+libsvm);
	        	
	        	if(j==0){
	        		validLabel=validMat.getLabel();
	        		testLabel=testMat.getLabel();
	        	}
		        
		        
	        	Params param = new Params() {
		            {
		            	put("booster","gbtree");
		                put("eta", 0.02);
		                put("scale_pos_weight", 8.7);//8.7
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
		                //put("max_delta_step", 1);//0 1-10
		            }
		        };
		        
		        //set round
		        int round = 1600;	        
		        //specify watchList
		        List<Map.Entry<String, DMatrix>> watchs =  new ArrayList<>();
		        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMat));
		        watchs.add(new AbstractMap.SimpleEntry<>("valid", validMat)); 
		        watchs.add(new AbstractMap.SimpleEntry<>("test", testMat));  
		        
		         
		        Booster booster = Trainer.train(param, trainMat, round, watchs, null, null);
		      //predict
//		        trainPredicts[j] = booster.predict(trainMat);  
		        validPredicts[j] = booster.predict(validMat); 
		        testPredicts[j] = booster.predict(testMat);  
		        
		        AUCEval eval = new AUCEval();	
		        float testAuc = eval.evalAUC(testPredicts[j], testMat.getLabel());
		        System.out.println("!!!!!BaseClassifer\t"+j+"\tauc=" + testAuc+"\n");
		        Base_fuction.Out_file(outResultFile, "BaseClassifer\t"+j+"\tauc=" + testAuc+"\n", true);
		        
	        }
	        
	        double baseClassiferWeight[]=PSO_feature_weight.excute(classifierNum, validPredicts,validLabel);
	        float testEnsemblePredicts[][]=Base_fuction.caculate_ensemble_result(testPredicts, baseClassiferWeight);
	        AUCEval eval = new AUCEval();	
	        aucScore[i] = eval.eval(testEnsemblePredicts, testLabel);
	        System.out.println("Ensemble auc=" + aucScore[i]+"\n");
	        Base_fuction.Out_file(outResultFile, "Ensemble auc=" + aucScore[i]+"\n", true);
	        
	        
			
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
