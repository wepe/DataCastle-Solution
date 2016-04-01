package edu.cqupt.xdata.model;

import java.io.IOException;

import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.demo.util.Params;
import org.dmlc.xgboost4j.demo.util.Txt2LibSvm;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.util.XGBoostError;

public class FeatureCrossValidation {
	
	static String arff=".arff";
	static String txt=".txt";
	static String csv=".csv";
	static String libsvm=".libsvm.txt";
	
	 public static void main(String[] args) throws IOException, XGBoostError {
         //load train mat
//		 String  inputTrainFilePath ="C:\\Users\\dell\\Desktop\\competeDianzi\\data\\all\\";
		 String  inputTrainFilePath ="";
//		 String  filename = "train_final_nouid";	
		 String  filename = args[0];
		 final double eta = Double.parseDouble(args[1]);
		 final double max_depth = Double.parseDouble(args[2]);
		 double round = Double.parseDouble(args[3]);
		         
//	    Txt2LibSvm.uci2Libsvm(inputTrainFilePath+filename+csv,inputTrainFilePath+filename+libsvm);		 
        DMatrix trainMat = new DMatrix(inputTrainFilePath+filename+libsvm);        
        //set params
        Params param = new Params() {
            {   put("booster","gbtree");
                put("eta", eta);//0.02
                put("scale_pos_weight", 8.7);//8.7
                put("lambda", 700);
                put("subsample", 0.7);
                put("colsample_bytree", 0.30);
                put("min_child_weight", 1);//5
                put("max_depth", max_depth);//8
                put("silent", 1);
                put("nthread", 40);
                put("objective", "binary:logistic");
                put("gamma", 0);
                put("eval_metric", "auc");
//                put("max_delta_step", 1);//0 1-10
                
            }
        };
        
        //do 5-fold cross validation
        int rounds = (int) round;//1520
        int nfold = 10;
        //set additional eval_metrics
        String[] metrics = null;        
        String[] evalHist = Trainer.crossValiation(param, trainMat, rounds, nfold, metrics, null, null);
    }	 
	 

}
