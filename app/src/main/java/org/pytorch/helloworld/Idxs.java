package org.pytorch.helloworld;

//public class Idxs implements Comparable<Idxs> {

//    private int key;
//    private float val;
//    public Idxs(int key, float val) {
//        this.key = key;
//        this.val = val;
//    }
//
//    public int getIndex() {
//        return key;
//    }
//    public float getval() {
//        return val;
//    }
//
//    @Override
//    public int compareTo(Idxs idxs) {
//        return 0;
//    }
//    private Integer a;
//    private Float b;
//    public Integer getA() {
//        return a;
//    }
//    public void setA(Integer a) {
//        this.a = a;
//    }
//    public Float getB() {
//        return b;
//    }
//    public void setB(Float b) {
//        this.b = b;
//    }
//}

class Idxs implements Comparable<Idxs>{
    private int key;
    private float val;

    public Idxs(int key, float val) {
        super();
        this.key = key;
        this.val = val;
    }

    public int getIndex() {
        return key;
    }
    public float getVal() {
        return val;
    }

    public void setIndex(int key){
        this.key = key;
    }
    public void setVal(float val) {

        this.val = val;
    }
    @Override
    public String toString() {
        return "Person [name=" + key + ", age=" + val + "]";
    }
    @Override
    public int compareTo(Idxs idxs) {
        //return this.getVal()-idxs.getVal();//排序 升序
        //return idxs.getVal()-this.getVal();//排序 降序
        return new Float(this.getVal()).compareTo(new Float(idxs.getVal()));
    }
}