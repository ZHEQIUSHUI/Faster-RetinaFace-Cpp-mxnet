#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mxnet-cpp/MxNetCpp.h>
#include <math.h>
#include <fstream>

using namespace mxnet::cpp;
using namespace std;
using namespace cv;
class CRect2f {
public:
	CRect2f(float x1, float y1, float x2, float y2) {
		val[0] = x1;
		val[1] = y1;
		val[2] = x2;
		val[3] = y2;
	}

	float& operator[](int i) {
		return val[i];
	}

	float operator[](int i) const {
		return val[i];
	}

	float val[4];

	void print() {
		printf("rect %f %f %f %f\n", val[0], val[1], val[2], val[3]);
	}
};

class Anchor {
public:
	Anchor() {
	}

	~Anchor() {
	}

	bool operator<(const Anchor &t) const {
		return score < t.score;
	}

	bool operator>(const Anchor &t) const {
		return score > t.score;
	}

	float& operator[](int i) {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	float operator[](int i) const {
		assert(0 <= i && i <= 4);

		if (i == 0)
			return finalbox.x;
		if (i == 1)
			return finalbox.y;
		if (i == 2)
			return finalbox.width;
		if (i == 3)
			return finalbox.height;
	}

	cv::Rect2f anchor; // x1,y1,x2,y2 
	float reg[4]; // offset reg
	cv::Point center; // anchor feat center
	float score; // cls score
	std::vector<cv::Point2f> pts; // pred pts

	cv::Rect2f finalbox; // final box res
	cv::Rect2f bbox;
	void print() {
		printf("finalbox %f %f %f %f, score %f\n", finalbox.x, finalbox.y, finalbox.width, finalbox.height, score);
		printf("landmarks ");
		for (int i = 0; i < pts.size(); ++i) {
			printf("%f %f, ", pts[i].x, pts[i].y);
		}
		printf("\n");
	}
};


class Face
{
public:
	Rect bbox;
	float score;
	vector<Point2f> landmarks;
	Face()
	{

	}
	~Face()
	{
		landmarks.swap(vector<Point2f>());
	}
};


class AnchorCfg {
public:
	std::vector<float> SCALES;
	std::vector<float> RATIOS;
	int BASE_SIZE;

	AnchorCfg() {}
	~AnchorCfg() {}
	AnchorCfg(const std::vector<float> s, const std::vector<float> r, int size) {
		SCALES = s;
		RATIOS = r;
		BASE_SIZE = size;
	}
};

class AnchorGenerator
{
	void _ratio_enum(const CRect2f& anchor, const std::vector<float>& ratios, std::vector<CRect2f>& ratio_anchors) {
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		ratio_anchors.clear();
		float sz = w * h;
		for (int s = 0; s < ratios.size(); ++s) {
			float r = ratios[s];
			float size_ratios = sz / r;
			float ws = std::sqrt(size_ratios);
			float hs = ws * r;
			ratio_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
				y_ctr - 0.5 * (hs - 1),
				x_ctr + 0.5 * (ws - 1),
				y_ctr + 0.5 * (hs - 1)));
		}
	}

	void _scale_enum(const std::vector<CRect2f>& ratio_anchor, const std::vector<float>& scales, std::vector<CRect2f>& scale_anchors) {
		scale_anchors.clear();
		for (int a = 0; a < ratio_anchor.size(); ++a) {
			CRect2f anchor = ratio_anchor[a];
			float w = anchor[2] - anchor[0] + 1;
			float h = anchor[3] - anchor[1] + 1;
			float x_ctr = anchor[0] + 0.5 * (w - 1);
			float y_ctr = anchor[1] + 0.5 * (h - 1);

			for (int s = 0; s < scales.size(); ++s) {
				float ws = w * scales[s];
				float hs = h * scales[s];
				scale_anchors.push_back(CRect2f(x_ctr - 0.5 * (ws - 1),
					y_ctr - 0.5 * (hs - 1),
					x_ctr + 0.5 * (ws - 1),
					y_ctr + 0.5 * (hs - 1)));
			}
		}

	}

	void bbox_pred(const CRect2f& anchor, const CRect2f& delta, cv::Rect2f& box) {
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		float dx = delta[0];
		float dy = delta[1];
		float dw = delta[2];
		float dh = delta[3];

		float pred_ctr_x = dx * w + x_ctr;
		float pred_ctr_y = dy * h + y_ctr;
		float pred_w = std::exp(dw) * w;
		float pred_h = std::exp(dh) * h;

		box = cv::Rect2f(pred_ctr_x - 0.5 * (pred_w - 1.0),
			pred_ctr_y - 0.5 * (pred_h - 1.0),
			pred_ctr_x + 0.5 * (pred_w - 1.0),
			pred_ctr_y + 0.5 * (pred_h - 1.0));
	}

	void landmark_pred(const CRect2f anchor, const std::vector<cv::Point2f>& delta, std::vector<cv::Point2f>& pts) {
		float w = anchor[2] - anchor[0] + 1;
		float h = anchor[3] - anchor[1] + 1;
		float x_ctr = anchor[0] + 0.5 * (w - 1);
		float y_ctr = anchor[1] + 0.5 * (h - 1);

		pts.resize(delta.size());
		for (int i = 0; i < delta.size(); ++i) {
			pts[i].x = delta[i].x*w + x_ctr;
			pts[i].y = delta[i].y*h + y_ctr;
		}
	}
public:
	
	// init different anchors
	int Init(int stride, const AnchorCfg& cfg, bool dense_anchor) {
		CRect2f base_anchor(0, 0, cfg.BASE_SIZE - 1, cfg.BASE_SIZE - 1);
		std::vector<CRect2f> ratio_anchors;
		// get ratio anchors
		_ratio_enum(base_anchor, cfg.RATIOS, ratio_anchors);
		_scale_enum(ratio_anchors, cfg.SCALES, preset_anchors);

		// save as x1,y1,x2,y2
		if (dense_anchor) {
			assert(stride % 2 == 0);
			int num = preset_anchors.size();
			for (int i = 0; i < num; ++i) {
				CRect2f anchor = preset_anchors[i];
				preset_anchors.push_back(CRect2f(anchor[0] + int(stride / 2),
					anchor[1] + int(stride / 2),
					anchor[2] + int(stride / 2),
					anchor[3] + int(stride / 2)));
			}
		}

		anchor_stride = stride;

		anchor_num = preset_anchors.size();
		return anchor_num;
	}

	
	int FilterAnchor(mxnet::cpp::NDArray& cls, mxnet::cpp::NDArray& reg, mxnet::cpp::NDArray& pts,bool use_landmarks,float cls_threshold, std::vector<Anchor>& result)
	{
		assert(cls.GetShape()[1] == anchor_num2);
		assert(reg.GetShape()[1] == anchor_num4);
		int pts_length = 0;
		if (use_landmarks)
		{
			assert(pts.GetShape()[1] % anchor_num == 0);
			pts_length = pts.GetShape()[1] / anchor_num / 2;
		}
		
		int w = cls.GetShape()[3];
		int h = cls.GetShape()[2];

		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				int id = i * w + j;
				for (int a = 0; a < anchor_num; ++a)
				{
					//std::cout<< j << "--------"<< i << "--------"<< id << "--------"<<cls.channel(anchor_num + a)[id]<<std::endl;
					if (cls.GetData()[(anchor_num + a) * h * w + id] >= cls_threshold) {
						//printf("cls %f\n", cls.channel(anchor_num + a)[id]);
						CRect2f box(j * anchor_stride + preset_anchors[a][0],
							i * anchor_stride + preset_anchors[a][1],
							j * anchor_stride + preset_anchors[a][2],
							i * anchor_stride + preset_anchors[a][3]);
						//printf("%f %f %f %f\n", box[0], box[1], box[2], box[3]);
						CRect2f delta(reg.GetData()[(a * 4 + 0) * h * w + id],
							reg.GetData()[(a * 4 + 1) * h * w + id],
							reg.GetData()[(a * 4 + 2) * h * w + id],
							reg.GetData()[(a * 4 + 3) * h * w + id]);

						Anchor res;
						res.anchor = cv::Rect_< float >(box[0], box[1], box[2], box[3]);
						bbox_pred(box, delta, res.finalbox);
						//printf("bbox pred\n");
						res.score = cls.GetData()[(anchor_num + a) * h * w + id];
						res.center = cv::Point(j, i);

						//printf("center %d %d\n", j, i);

						if (use_landmarks) {
							std::vector<cv::Point2f> pts_delta(pts_length);
							for (int p = 0; p < pts_length; ++p) {
								pts_delta[p].x = pts.GetData()[(a*pts_length * 2 + p * 2) * h * w + id];
								pts_delta[p].y = pts.GetData()[(a*pts_length * 2 + p * 2 + 1) * h * w + id];
							}
							//printf("ready landmark_pred\n");
							landmark_pred(box, pts_delta, res.pts);
							//printf("landmark_pred\n");
						}
						result.push_back(res);
					}
				}
			}
		}
		return 0;
	}

	AnchorGenerator() {};
	~AnchorGenerator() {};

private:
	std::vector<std::vector<Anchor>> anchor_planes; // corrspont to channels
	std::vector<int> anchor_size;
	std::vector<float> anchor_ratio;
	float anchor_step; // scale step
	int anchor_stride; // anchor tile stride
	int feature_w; // feature map width
	int feature_h; // feature map height
	std::vector<CRect2f> preset_anchors;
	int anchor_num; // anchor type num
};

class RetinaFace
{
	Context *Ctx = nullptr;
	Symbol sym_net;
	std::map<std::string, mxnet::cpp::NDArray> args;
	std::map<std::string, mxnet::cpp::NDArray> aux;
	
  mxnet::cpp::NDArray data2ndarray(mxnet::cpp::Context ctx, float * data, int batch_size, int num_channels, int height, int width)
  {
    mxnet::cpp::NDArray ret(mxnet::cpp::Shape(batch_size, num_channels, height, width), ctx, false);

    ret.SyncCopyFromCPU(data, batch_size * num_channels * height * width);

    ret.WaitToRead();  //mxnet::cpp::NDArray::WaitAll();

    return ret;
   }

public:
	bool use_landmarks = true;
	float nms_threshold = 0.4;
	RetinaFace(bool use_gpu) 
	{
		Ctx = use_gpu ? new Context(kGPU, 0) : new Context(kCPU, 0);
		for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
			int stride = _feat_stride_fpn[i];
			AnchorGenerator a;
			a.Init(stride, anchor_cfg[stride], false);
			ac.push_back(a);
		}
	}
	~RetinaFace() { delete Ctx; }
	void Loadmodel(String floder, String prefix)
	{
		sym_net = Symbol::Load(floder + "/" + prefix + "-symbol.json");
		std::map<std::string, mxnet::cpp::NDArray> params;
		NDArray::Load(floder + "/" + prefix + "-0000.params", nullptr, &params);
		for (const auto &k : params)
		{
			if (k.first.substr(0, 4) == "aux:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				aux[name] = k.second.Copy(*Ctx);
			}
			if (k.first.substr(0, 4) == "arg:")
			{
				auto name = k.first.substr(4, k.first.size() - 4);
				args[name] = k.second.Copy(*Ctx);
			}
		}
		// WaitAll is need when we copy data between GPU and the main memory
		mxnet::cpp::NDArray::WaitAll();
	}
	std::vector<Face> detect(Mat img,float threshold=0.8, float scale = 1.0)
	{
		vector<Face> faces;
		std::vector<Anchor> proposals;
		proposals.clear();

		Mat temp;
		if (scale != 1.0)
		{
			resize(img, temp, Size(), scale, scale);
		}
		else
		{
			img.copyTo(temp);
		}

		temp.convertTo(temp, CV_32FC3);
		Mat bgr[3];
		split(temp, bgr);

		Mat b_img = bgr[0];
		Mat g_img = bgr[1];
		Mat r_img = bgr[2];

		int len_img = temp.cols * temp.rows;
		float* data_img = new float[temp.channels()*len_img]; //rrrrr...ggggg...bbbbb...
		memcpy(data_img, r_img.data, len_img * sizeof(*data_img));
		memcpy(data_img + len_img, g_img.data, len_img * sizeof(*data_img));
		memcpy(data_img + len_img + len_img, b_img.data, len_img * sizeof(*data_img));

		NDArray data = data2ndarray(*Ctx, data_img, 1, 3, temp.rows, temp.cols);

		args["data"] = data;
		Executor *exec = sym_net.SimpleBind(*Ctx, args, map<string, NDArray>(), map<string, OpReqType>(), aux);
		exec->Forward(false);
		for (int _idx = 0; _idx < _feat_stride_fpn.size(); _idx++)
		{
			int idx = _idx * 3;
			vector<mx_float> cls_data;
			vector<uint> cls_shape = exec->outputs[idx].GetShape();
			exec->outputs[idx].SyncCopyToCPU(&cls_data, exec->outputs[idx].Size());
			idx++;
			vector<mx_float> reg_data;
			vector<uint> reg_shape = exec->outputs[idx].GetShape();
			exec->outputs[idx].SyncCopyToCPU(&reg_data, exec->outputs[idx].Size());
			idx++;
			NDArray cls(cls_data, Shape(cls_shape), Context::cpu());
			NDArray reg(reg_data, Shape(reg_shape), Context::cpu());
			NDArray pts;
			vector<mx_float> pts_data;
			vector<uint> pts_shape = exec->outputs[idx].GetShape();
			if (use_landmarks)
			{
				exec->outputs[idx].SyncCopyToCPU(&pts_data, exec->outputs[idx].Size());
				pts = NDArray(pts_data, Shape(pts_shape), Context::cpu());
			}
			ac[_idx].FilterAnchor(cls, reg, pts, use_landmarks, threshold, proposals);
		}
		std::vector<Anchor> result;
		nms_cpu(proposals, nms_threshold, result);
		for (size_t i = 0; i < result.size(); i++)
		{
			Face face;
			face.score = result[i].score;
			float x0 = result[i].finalbox.x / scale;
			float y0 = result[i].finalbox.y / scale;
			float x1 = result[i].finalbox.width / scale;
			float y1 = result[i].finalbox.height / scale;
			Point tl(x0, y0);
			Point br(x1, y1);
			face.bbox= Rect(tl, br);
			//result[i].bbox = Rect(tl, br);
			for (size_t j = 0; j < result[i].pts.size(); j++)
			{
				face.landmarks.push_back(Point2f(result[i].pts[j].x / scale, result[i].pts[j].y /= scale));
			}
			faces.push_back(face);
		}
		delete[] data_img;
		data_img = nullptr;
		delete exec;
		exec = nullptr;
		return faces;
	}
private:
	std::vector<AnchorGenerator> ac;// (_feat_stride_fpn.size());
	
	std::vector<int> _feat_stride_fpn = { 32, 16, 8 };
	std::map<int, AnchorCfg> anchor_cfg = {
		{ 32, AnchorCfg(std::vector<float>{32,16}, std::vector<float>{1}, 16) },
		{ 16, AnchorCfg(std::vector<float>{8,4}, std::vector<float>{1}, 16) },
		{ 8,AnchorCfg(std::vector<float>{2,1}, std::vector<float>{1}, 16) }
	};
	
};


