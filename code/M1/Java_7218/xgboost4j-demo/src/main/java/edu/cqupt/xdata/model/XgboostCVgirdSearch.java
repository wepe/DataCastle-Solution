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
public class XgboostCVgirdSearch {
	
	static String arff=".arff";
	static String txt=".txt";
	static String csv=".csv";
	static String libsvm=".libsvm.txt";
	
	public static final int BUFSIZE = 1024 * 8; 
	
	public static void main(String[] args) throws IOException, XGBoostError {
		
		//参数
		int nFlod = 10;
		
		
		 int max_delta_step[] = {1,2,3,4,5,6,7,8,9,10};//10	     
	     int round[]={110,120,130,140,150,160,170,180,190,200};//10	     
	     double scale_pos_weight[]={8.0,8.5,9.0,9.5};//4
	     int max_depth[] ={5,6,7,8,9,10};//6
	     double eta[] ={0.005,0.01,0.015,0.02};//4
	     int classifierNum = max_depth.length*round.length*max_delta_step.length*scale_pos_weight.length*eta.length;		
		//原始整个训练集数据
//		String inputFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\20160121\\";
	     String inputFilePath ="";
		String filename = args[0];	
	     String outResultFile="process_search.txt";
	     String outResultFile2="process_best.txt";
	     
	     Base_fuction.Out_file(outResultFile, "", false);
	     Base_fuction.Out_file(outResultFile2, "", false);
//		//将csv 转为arff
		Txt2Arff.txt2Arff(inputFilePath+filename+".csv", inputFilePath+filename+".arff");
		
		//将arff文件分折
		Base_fuction source = new Base_fuction(filename,inputFilePath);		
		source.partion_CV(0, nFlod, source.GetOri_instances(), filename, inputFilePath+"cv\\");
		
		
		//将每折的训练集测试集读入内存
		DMatrix trainMatALL [] = new DMatrix[nFlod];
		DMatrix testMatALL[] = new DMatrix[nFlod];
		for(int i = 0;i < nFlod;i++){
			System.out.println("Begin read data"+i+" cv");
			//将.arff 转为libsvm格式			
			String trainfile=filename+"_CV"+i+"_train";
			String testfile=filename+"_CV"+i+"_test";
			Txt2LibSvm.uci2Libsvm(inputFilePath+"cv\\"+trainfile+arff,inputFilePath+"cv\\"+trainfile+libsvm);
			Txt2LibSvm.uci2Libsvm(inputFilePath+"cv\\"+testfile+arff,inputFilePath+"cv\\"+testfile+libsvm);	
		    //load valid mat (svmlight format)
			trainMatALL[i] = new DMatrix(inputFilePath+"cv\\"+trainfile+libsvm);
			testMatALL[i] = new DMatrix(inputFilePath+"cv\\"+testfile+libsvm);			
		}
		  System.out.println("Read data finished");
		
		
		
		float maxAucSore = 0f; 
		
		
		for(int nummax_delta_step = 0;nummax_delta_step <max_delta_step.length;nummax_delta_step++){
			for(int numround = 0;numround <round.length; numround++){
				for(int numscale_pos_weight = 0; numscale_pos_weight < scale_pos_weight.length;numscale_pos_weight++){
					for(int nummax_depth = 0;nummax_depth <max_depth.length;nummax_depth++){
						for(int numeta = 0;numeta < eta.length;numeta++){
							
					int j = numeta+nummax_depth*eta.length+numscale_pos_weight*max_depth.length*eta.length+numround*(scale_pos_weight.length*max_depth.length*eta.length+numround)
							+nummax_delta_step*(round.length*scale_pos_weight.length*max_depth.length*eta.length);
							
							//set params
				        	final double eta1=eta[numeta];
				        	final double scale_pos_weight1=scale_pos_weight[numscale_pos_weight];
				        	final double max_depth1=max_depth[nummax_depth];
				        	final double max_delta_step1=max_delta_step[nummax_delta_step];
				        	  //set round
					        int round1 = round[numround]*10;
				        	
				        	Params param = new Params() {
					            {
					                put("eta", eta1);
					                put("scale_pos_weight", scale_pos_weight1);//8.7
					                put("lambda", 700);	               	               
					                put("min_child_weight", 5);
					                put("max_depth", max_depth1);
					                put("silent", 1);
					                put("nthread", 40);
					                put("subsample", 0.7);
					                put("colsample_bytree", 0.80);
					                put("objective", "binary:logistic");
					                put("gamma", 0);
					                put("eval_metric", "auc");
					                put("max_delta_step", max_delta_step1);//0 1-10
					            }
					        };       
					      	
					        
					        float aucScore[] = new float[nFlod];
					        
					        for(int cvNum = 0;cvNum < nFlod;cvNum++){
						        //specify watchList
						        List<Map.Entry<String, DMatrix>> watchs =  new ArrayList<>();
//						        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMatALL[cvNum]));
//						        watchs.add(new AbstractMap.SimpleEntry<>("test", testMatALL[cvNum]));  
						        
						         
						        Booster booster = Trainer.train(param, trainMatALL[cvNum], round1, watchs, null, null);
						      //predict
						       float testPredicts[][] = booster.predict(testMatALL[cvNum]);  
						        
						        AUCEval eval = new AUCEval();	
						        aucScore[cvNum] = eval.evalAUC(testPredicts, testMatALL[cvNum].getLabel());
					        	
					        	
					        }         	        
                          
					        float baseresAuc =  calAvgAuc(aucScore);
					        if(baseresAuc > maxAucSore){
					        	maxAucSore = baseresAuc;
					        	System.out.println("-->bestClassifer\t"+j+"\tauc=" + baseresAuc+"\n");
					        	   String res1 = "-->bestClassifer\t"+j+"\tauc=" + baseresAuc+"\n"
								    		+eta[numeta]+"\t"+scale_pos_weight[numscale_pos_weight]+"\t"+
								    		max_depth[nummax_depth]+"\t"+max_delta_step[nummax_delta_step]+"\t"+round[numround]+"\n";
									Base_fuction.Out_file(outResultFile2, res1, true); 
					        	
					        }
					       System.out.println("!!!!!BaseClassifer\t"+j+"\tauc=" + baseresAuc+"\n");					       

							
						    String res = "!!!!!BaseClassifer\t"+j+"\tauc=" + baseresAuc+"\n"
						    		+eta[numeta]+"\t"+scale_pos_weight[numscale_pos_weight]+"\t"+
						    		max_depth[nummax_depth]+"\t"+max_delta_step[nummax_delta_step]+"\t"+round[numround]+"\n";
							Base_fuction.Out_file(outResultFile, res, true);    
					       
							
							
						}
					}
					
				}
			}
			
		}		
		
		

		
	}
	
	public static float calAvgAuc(float [] aucScore){
		
		float sum = 0f;
		for(int i = 0;i < aucScore.length; i++){
			sum+= aucScore[i];
		}		
		return sum/aucScore.length;
	}
	
 
	

}
