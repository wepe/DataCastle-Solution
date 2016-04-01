package org.dmlc.xgboost4j.demo.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.IEvaluation;
import org.dmlc.xgboost4j.util.XGBoostError;

import zr.until.AUC.Tuple2D;

public class AUCEval implements IEvaluation {
	private static final Log logger = LogFactory.getLog(CustomEval.class);

    String evalMetric = "auc";
        
    @Override
    public String getMetric() {
        return evalMetric;
    }

    @Override
    public float eval(float[][] predicts, DMatrix dmat) {
        float error = 0f;
        float[] labels;
        try {
            labels = dmat.getLabel();
        } catch (XGBoostError ex) {
            logger.error(ex);
            return -1f;
        }
        int nrow = predicts.length;
        
    	HashMap<Integer, String> map_ans_0=new HashMap<>();//存答案0
		HashMap<Integer, String> map_ans_1=new HashMap<>();//存答案1
		HashMap<Integer, String> map_test=new HashMap<>();//存测试数据
		String line="";
		//先读入answer数据
		for(int i = 0; i < labels.length;i++){
			 if (labels[i] == 0f) {
				 map_ans_0.put(i,labels[i]+"");
			}else {
				map_ans_1.put(i,labels[i]+"");
			}
		}

		//读入test数据
		for(int i = 0; i < nrow;i++){
//			System.out.println(i+" "+predicts[i][0]+"");
			map_test.put(i,predicts[i][0]+"");
		}

		//组成样本对
		ArrayList<String> list=new ArrayList<>();
		for(Integer key0:map_ans_0.keySet())
			for(Integer  key1:map_ans_1.keySet())
			{
				line=key0+","+key1;
				list.add(line);
			}
		//计算得分
		double score=0;
		for(String s:list)
		{
			String temp[]=s.split(",");
			Integer neg=Integer.parseInt(temp[0]);
			Integer pos=Integer.parseInt(temp[1]);
//			System.out.println(map_test.get(pos));
//			System.out.println(map_test.get(neg));
			double p_score=Double.parseDouble(map_test.get(pos));
			double n_score=Double.parseDouble(map_test.get(neg));
			if (p_score>n_score) {
				score+=1;
			}else if (p_score==n_score) {
				score+=0.5;
			}
		}
		
//		System.out.println(score/list.size());
		return (float) (score/list.size());        

    }

    public float eval(float[][] predicts, float[] labels) {
        float error = 0f;
        int nrow = predicts.length;
        
    	HashMap<Integer, String> map_ans_0=new HashMap<>();//存答案0
		HashMap<Integer, String> map_ans_1=new HashMap<>();//存答案1
		HashMap<Integer, String> map_test=new HashMap<>();//存测试数据
		String line="";
		//先读入answer数据
		for(int i = 0; i < labels.length;i++){
			 if (labels[i] == 0f) {
				 map_ans_0.put(i,labels[i]+"");
			}else {
				map_ans_1.put(i,labels[i]+"");
			}
		}

		//读入test数据
		for(int i = 0; i < nrow;i++){
//			System.out.println(i+" "+predicts[i][0]+"");
			map_test.put(i,predicts[i][0]+"");
		}

		//组成样本对
		ArrayList<String> list=new ArrayList<>();
		for(Integer key0:map_ans_0.keySet())
			for(Integer  key1:map_ans_1.keySet())
			{
				line=key0+","+key1;
				list.add(line);
			}
		//计算得分
		double score=0;
		for(String s:list)
		{
			String temp[]=s.split(",");
			Integer neg=Integer.parseInt(temp[0]);
			Integer pos=Integer.parseInt(temp[1]);
//			System.out.println(map_test.get(pos));
//			System.out.println(map_test.get(neg));
			double p_score=Double.parseDouble(map_test.get(pos));
			double n_score=Double.parseDouble(map_test.get(neg));
			if (p_score>n_score) {
				score+=1;
			}else if (p_score==n_score) {
				score+=0.5;
			}
		}
		
//		System.out.println(score/list.size());
		return (float) (score/list.size());        

    }
    
    
 
	public float evalAUC(float[][] predicts, float[] labels){
		ArrayList<Tuple2D> data = new ArrayList<Tuple2D>();
		double truth = 0, class1;
		for(int i = 0;i < labels.length;i++){
			if( labels[i] == 0f)
        		truth = 1f;
        	else if (labels[i] == 1f)
        		truth = 0f;
			class1 = 1-predicts[i][0];
			data.add(new Tuple2D(truth, class1));
		}
	    Collections.sort(data);
	    double auc = sortingAUC(data);

//	    puts("auc: %f", auc);
		
		return (float) auc;
	}
	
	/**
	 * requires input to be sorted by class1, though it doesn't check
	 * @param data
	 * @return
	 */
	public static double sortingAUC(ArrayList<Tuple2D> data){
		double P = sumTruth(data);
		double N = data.size() - P;
		double n = data.size();
		//puts("sortingAUC: P %f, N %f, n %f", P, N, n);
		
		double lastpred = data.get(0).pred - 1.;
		double lastx = 0, lasty = 0;
		double x, y;
		double tp = 0, fp = 0;
		
		double auc = 0;
		for(Tuple2D tuple : data){
			// add point <fp/N, tp/P> to roc curve
			if( tuple.pred != lastpred){
				x = fp / N;
				y = tp / P;
				auc = auc + 0.5 * (lasty + y) * (x - lastx); // trap area
				lastx = x;
				lasty = y;
				lastpred = tuple.pred;
			}
			if( tuple.truth > 0 )
				tp++;
			else
				fp++;
		}
		
		return auc;
	}
	
	public static double sumTruth(ArrayList<Tuple2D> data){
		double sum = 0d;
		for(Tuple2D tuple : data)
			sum += tuple.truth;
		return sum;
	}

	
	
	public static class Tuple2D implements Comparable<Tuple2D> {
		public double truth, pred;
		public Tuple2D(double truth, double class1){ this.truth = truth; this.pred = class1; }
		
		@Override
		public int compareTo(Tuple2D that) {
			return -1 * Double.compare(this.pred, that.pred);
		}
	}
	
	public static void puts(String s){System.out.println(s);}
	public static void puts(String format, Object... args){ puts(String.format(format, args)); }

}
