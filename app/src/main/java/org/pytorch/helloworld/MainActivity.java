package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Color;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
//import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import androidx.appcompat.app.AppCompatActivity;

import static java.lang.Thread.sleep;

public class MainActivity extends AppCompatActivity {

  public void getPicturePixel(Bitmap bitmap) {

    int width = bitmap.getWidth();
    int height = bitmap.getHeight();

    // 保存所有的像素的数组，图片宽×高
    int[] pixels = new int[width * height];

    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

    for (int i = 0; i < pixels.length; i++) {
      int clr = pixels[i];
      int red = (clr & 0x00ff0000) >> 16; // 取高两位
//      Log.d("tag", "r=" + red);
      int green = (clr & 0x0000ff00) >> 8; // 取中两位
      int blue = clr & 0x000000ff; // 取低两位
//      Log.d("tag", "r=" + red + ",g=" + green + ",b=" + blue);
    }

  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmap = null;
    Module module = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      bitmap = BitmapFactory.decodeStream(getAssets().open("our_23.png")).copy(Bitmap.Config.ARGB_8888, true);
      Log.d("bitmap", "------------------width: " + bitmap.getWidth()+"---------hight: "+bitmap.getHeight());
      //打印三个像素点
//      for (int i = 10; i <= 20 ; i++){
//        int red = (bitmap.getPixel(0,i) & 0x00ff0000) >> 16; // 取高两位
////      Log.d("tag", "r=" + red);
//        int green = (bitmap.getPixel(0,i) & 0x0000ff00) >> 8; // 取中两位
//        int blue = bitmap.getPixel(0,i) & 0x000000ff; // 取低两位
//        Log.d("rgb++++++++++++++", "r=" + red + ",g=" + green + ",b=" + blue);
//      }
//      getPicturePixel(bitmap);
      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      module = Module.load(assetFilePath(this, "ssd_model.pt"));

      ImageView imageView = findViewById(R.id.image);

      TextView textView = findViewById(R.id.text);
//    } catch (IOException e) {
//      Log.e("PytorchHelloWorld", "Error reading assets", e);
//      finish();
//    }


    // preparing input tensor
    Bitmap bitmap1 = Bitmap.createScaledBitmap(bitmap, 300, 300, false);
      Log.d("bitmap1", "------------------width: " + bitmap1.getWidth()+"---------hight: "+bitmap1.getHeight());
//      for (int i = 0; i < 10 ; i++){
//        int red = (bitmap1.getPixel(i,0) & 0x00ff0000) >> 16; // 取高两位
////      Log.d("tag", "r=" + red);
//        int green = (bitmap1.getPixel(i,0) & 0x0000ff00) >> 8; // 取中两位
//        int blue = bitmap1.getPixel(i,0) & 0x000000ff; // 取低两位
//        Log.i("resize-rgb++++++++++++", "r=" + red + ",g=" + green + ",b=" + blue);
//      }
//    Log.i("bitmap1", "------"+bitmap1.toString());
//    float[] mean = {127, 127, 127};
//    float[] std = {128, 128, 128};
//    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap1,mean, std);
    Tensor inputTensor = imageUtil.bitmapToFloat32Tensor(bitmap1,imageUtil.TORCHVISION_NORM_MEAN_RGB, imageUtil.TORCHVISION_NORM_STD_RGB);
    Log.i("inputTensor", "------"+inputTensor);
    // running the model
    Log.i("output", "------"+module.forward(IValue.from(inputTensor)).toTuple()[0].toTensor());
//    Log.i("output", "------"+IValue.tupleFrom(module.forward(IValue.from(inputTensor)).toTuple()).toTensorList());
    final Tensor confidence = module.forward(IValue.from(inputTensor)).toTuple()[0].toTensor();
    final Tensor location = module.forward(IValue.from(inputTensor)).toTuple()[1].toTensor();
    // getting tensor content as java array of floats
    final float[] scores = confidence.getDataAsFloatArray();
    final float[] boxes = location.getDataAsFloatArray();

    Log.i("score","--------------------"+ Arrays.toString(scores));
      Log.i("score","--------------------"+ scores.length);
//    Log.i("boxes","--------------------"+boxes.toString());

    List<Float> bbox_max_scores = new ArrayList<>();
    List<Integer> bbox_max_scores_classes = new ArrayList<>();//0表示apple，1表示banana
    for (int i = 0; i < scores.length; i += 3) {
      int maxindx = 0;
      float maxscore = scores[i];
      for (int j = 1; j < 3; j++) {
        if (maxscore < scores[i + j]) {
          maxscore = scores[i + j];
          maxindx = j;
        }
      }
      bbox_max_scores.add(maxscore);
      bbox_max_scores_classes.add(maxindx);
    }
//    Log.i("bbox_max_scores","------"+bbox_max_scores.size());
//    Log.i("bbox_max_scores_classes","------"+bbox_max_scores_classes.size());

//      Log.i("bbox_max_scores","------"+bbox_max_scores.toString());
    //获取元素非0的索引，存入no_zero_index列表，用于获取对应boxes值
    final List<Integer> no_zero_index = new ArrayList<>();
    for (int k = 0; k <= bbox_max_scores_classes.size() - 1; k++) {
//            System.out.println(i);
      if (bbox_max_scores_classes.get(k) != 0) {
        no_zero_index.add(k);
      }
    }
//    Log.i("no_zero_index","------"+no_zero_index.size());
//    Log.i("no_zero_index","------"+no_zero_index.toString());
//        ArrayList<Integer> clazs = new ArrayList<Integer>();
//        ArrayList<Double> confs = new ArrayList<Double>();
// 获取元素非零的置信度，存入no_zeros_conf列表,获取非零类别存入no_zero_clazs
    final List<Float> no_zero_conf = new ArrayList<>();
    final List<Integer> no_zero_clazs = new ArrayList<>();
    for (int item : no_zero_index) {
      int claz = bbox_max_scores_classes.get(item);
      no_zero_clazs.add(claz);
//            System.out.println(keys.get((Integer) item));
    }
    for (int item : no_zero_index) {
      float conf = bbox_max_scores.get(item);
      no_zero_conf.add(conf);
//            System.out.println(keys.get((Integer) item));
    }

//      Log.i("no_zero_conf","------"+no_zero_conf.size());
//      Log.i("no_zero_clazs","------"+no_zero_clazs.size());
//      Log.i("no_zero_clazs","------"+no_zero_clazs.toString());
//获取对应的bbox信息，先将boxes转化为二维数组，再获取对应的bbox信息，相当于过滤掉背景边框
    float dim2_boxes[][] = TwoArry(boxes);
    List<float[]> no_zero_boxes = new ArrayList<>();
    for (int index : no_zero_index) {
      no_zero_boxes.add(dim2_boxes[index]);
    }

    //将boxes信息转化为float[][] bboxes
    final float[][] bboxes = no_zero_boxes.toArray(new float[no_zero_boxes.size()][]);
      Log.i("bboxes","------------------------"+bboxes[1].getClass());

//    Log.i("bboxes","------------------------"+bboxes.length);
    Log.i("bboxes","-------------------------"+Arrays.deepToString(bboxes));
    //极大值抑制，返回置信度最大的边框的位置索引
    List<Integer> pickList = single_class_non_max_suppression(bboxes, no_zero_conf);
    //根据索引获取最终的边框信息，类别信息，置信度信息
      Log.i("pickList","------------------------"+pickList.size());
      Log.i("pickList","------------------------"+pickList.toString());
    final List<float[]> last_bbox_list = new ArrayList<>();
    final List<Float> last_conf_list = new ArrayList<>();
    final List<Integer> last_claz_list = new ArrayList<>();
    for (int item : pickList) {
//      Log.i("bboxes[item]","------------------------"+bboxes[item].getClass());
      last_bbox_list.add(bboxes[item]);
    }
    for (int item : pickList) {
      float conf = no_zero_conf.get(item);
      last_conf_list.add(conf);
//            System.out.println(keys.get((Integer) item));
    }
//      System.out.println(last_conf_list.getClass().getName());
    for (int item : pickList) {
      int claz = no_zero_clazs.get(item);
      last_claz_list.add(claz);
//            System.out.println(keys.get((Integer) item));
    }

      Log.i("last_bbox_list","------------------------"+last_conf_list.get(1).getClass().getName());
//      Log.i("last_bbox_list","------------------------"+last_bbox_list.size());
//      Log.i("last_conf_list","-------------------------"+last_conf_list.size());
//      Log.i("last_bbox_list","------------------------"+Arrays.deepToString(last_bbox_list));
    float[][] last_bbox_arr = last_bbox_list.toArray(new float[last_bbox_list.size()][]);
      Log.i("last_bbox_list","------------------------"+Arrays.deepToString(last_bbox_arr));
    Float[] last_conf_arr = (Float[])last_conf_list.toArray(new Float[last_conf_list.size()]);
    Integer[] last_claz_arr = (Integer[]) last_claz_list.toArray(new Integer[last_claz_list.size()]);


//    //获取类名，暂时还有点问题
//    String className = ImageNetClasses.IMAGENET_CLASSES[1];
//
//    // showing className on UI
//    TextView textView = findViewById(R.id.text);
//    textView.setText(className);

        // Data format conversion takes too long
        // Log.d("inputData", Arrays.toString(inputData));
      long start = System.currentTimeMillis();
      // get predict result
//        float[] result = mobileNetssd.Detect(input_bmp);
      // time end
      long end = System.currentTimeMillis();
      Log.d("hahaha", "origin predict result:" + Arrays.toString(last_bbox_arr));
      long time = end - start;
      Log.d("result length", "length of result: " + String.valueOf(last_bbox_arr.length));
      // show predict result and time
      // float[] r = get_max_result(result);

      String show_text = "result：" + Arrays.toString(last_bbox_arr) + "\nname：" + Arrays.toString(last_claz_arr) + "\nprobability：" + Arrays.toString(last_conf_arr) + "\ntime：" + time + "ms" ;
//      textView.setText(show_text);

      // 画布配置
      Canvas canvas = new Canvas(bitmap);
      //图像上画矩形
      Paint paint = new Paint();
      paint.setColor(Color.RED);
      paint.setStyle(Paint.Style.STROKE);//不填充
      paint.setStrokeWidth(5); //线的宽度


//        float get_finalresult[][] = TwoArry(result);
//        Log.d("zhuanhuan",get_finalresult+"");
      int object_num = 0;
      int num = last_bbox_arr.length;// number of object
      List<String> class_name = new ArrayList<>(Arrays.asList("background","zhoukai","fangningdan"));

//        String[] class_name1 = {"zhoukai","fangningdan"};
      //continue to draw rect
      for(object_num = 0; object_num < num; object_num++){
//          Log.d("lsdgkjslg", "haha :" + Arrays.toString(get_finalresult));
        // 画框
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);//不填充
        paint.setStrokeWidth(5); //线的宽度
        canvas.drawRect(last_bbox_arr[object_num][0] * bitmap.getWidth(), last_bbox_arr[object_num][1] * bitmap.getHeight(),
                last_bbox_arr[object_num][2] * bitmap.getWidth(), last_bbox_arr[object_num][3] * bitmap.getHeight(), paint);
        Log.i("last_bbox_arr", "---------------left:" + last_bbox_arr[object_num][0] * bitmap1.getWidth());
        Log.i("last_bbox_arr", "---------------top:" + last_bbox_arr[object_num][1] * bitmap1.getWidth());
        Log.i("last_bbox_arr", "---------------right:" + last_bbox_arr[object_num][2] * bitmap1.getWidth());
        Log.i("last_bbox_arr", "---------------bottom:" + last_bbox_arr[object_num][3] * bitmap1.getWidth());
        paint.setColor(Color.YELLOW);
        paint.setStyle(Paint.Style.FILL);//不填充
        paint.setStrokeWidth(1); //线的宽度

        Log.i("text", "---------------text:" + class_name.get(2));
        Log.i("text", "---------------text:" + class_name.get(last_claz_arr[object_num]));
        Log.i("x", "---------------x:" + last_bbox_arr[object_num][0]*bitmap.getWidth());
        Log.i("y", "---------------y:" + last_bbox_arr[object_num][1]*bitmap.getHeight());
        canvas.drawText(class_name.get((int)last_claz_arr[object_num]) + "\n" + last_conf_arr[object_num],
                last_bbox_arr[object_num][0]*bitmap.getWidth(),last_bbox_arr[object_num][1]*bitmap.getHeight(),paint);
        }

          //往图上画东西
      imageView.setImageBitmap(bitmap);
    } catch (Exception e) {
      e.printStackTrace();
    }


    // showing image on UI


  }
  //    极大值抑制
  public static List<Integer> single_class_non_max_suppression ( float[][] bboxes, List<Float > confidences){
    if (bboxes.length == 0)
      return null;
    List<Integer> conf_keep_idx = new ArrayList<>();//保存置信度大于CONF_THRESH的元素的下标
    List<Float> new_confidences = new ArrayList<>();//保存置信度大于CONF_THRESH的元素的值

    for (int i = 0; i < confidences.size(); i++) {
      float confidence = confidences.get(i);
      if (confidence > 0.3) {
        conf_keep_idx.add(i);
        new_confidences.add(confidence);
      }
    }
    if (conf_keep_idx.isEmpty())
      return null;
//    Log.i("conf_keep_idx","------------------------"+conf_keep_idx.size());
//    Log.i("new_confidences","------------------------"+new_confidences.size());
    int alive_idx_size = conf_keep_idx.size();
    List<Idxs> idxsList = new ArrayList<>();//将置信度与下标对应
    for (int i = 0; i < alive_idx_size; i++) {
      Idxs idxs = new Idxs(conf_keep_idx.get(i), new_confidences.get(i));
      idxsList.add(idxs);
    }

    //按score升序排列
    Collections.sort(idxsList);

    float overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax;
    float overlap_w, overlap_h, overlap_area, overlap_ratio;
    //取出得分最高的bbox，计算剩下的bbox与它的交并比iou，去掉大于iou_thresh的bbox
    List<Integer> pickList = new ArrayList<>();
    while (idxsList.size() > 0) {
      try {
        sleep(20);
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
      int last = idxsList.size() - 1;
      if (pickList.size() >= 3)//取置信度最高的NUM_DETECTIONS个结果
        break;
      int last_index = idxsList.get(last).getIndex(); //最大置信度的位置索引
      float last_area = (bboxes[last_index][2] - bboxes[last_index][0]) * (bboxes[last_index][3] - bboxes[last_index][1]);//area=(xmax-xmin)*(ymax-ymin)
      pickList.add(last_index);
      List<Idxs> idxs_to_remove = new ArrayList<>();//交并比过大需要移除的bbox
      for (int i = 0; i < last; i++) {
        int i_index = idxsList.get(i).getIndex();
        overlap_xmin = Math.max(bboxes[last_index][0], bboxes[i_index][0]);
        overlap_ymin = Math.max(bboxes[last_index][1], bboxes[i_index][1]);
        overlap_xmax = Math.min(bboxes[last_index][2], bboxes[i_index][2]);
        overlap_ymax = Math.min(bboxes[last_index][3], bboxes[i_index][3]);
        overlap_w = Math.max(0, overlap_xmax - overlap_xmin);
        overlap_h = Math.max(0, overlap_ymax - overlap_ymin);
        overlap_area = overlap_w * overlap_h;
        float i_area = (bboxes[i_index][2] - bboxes[i_index][0]) * (bboxes[i_index][3] - bboxes[i_index][1]);
        overlap_ratio = overlap_area / (last_area + i_area - overlap_area);//IOU
        if (overlap_ratio > 0.5)
          idxs_to_remove.add(idxsList.get(i));
      }
      idxs_to_remove.add(idxsList.get(last));
      idxsList.removeAll(idxs_to_remove);
    }
    return pickList;
  }
  //一维数组转化为二维数组
  public static float[][] TwoArry ( float[] inputfloat){
    int n = inputfloat.length;
    int num = inputfloat.length / 4;
    float[][] outputfloat = new float[num][4];
    int k = 0;
    for (int i = 0; i < num; i++) {
      int j = 0;

      while (j < 4) {
        outputfloat[i][j] = inputfloat[k];
        k++;
        j++;
      }

    }
    return outputfloat;
  }
  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
