/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"
#include "opencv2/opencv.hpp"
#if HAS_ZIPLIB
#include "zip.h"
#endif

#include <boost/thread.hpp>

using namespace dso;

inline int getdir(std::string dir, std::vector<std::string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL)
	{
		return -1;
	}

	while ((dirp = readdir(dp)) != NULL)
	{
		std::string name = std::string(dirp->d_name);

		if (name != "." && name != "..")
			files.push_back(name);
	}
	closedir(dp);

	std::sort(files.begin(), files.end());

	if (dir.at(dir.length() - 1) != '/')
		dir = dir + "/";
	for (unsigned int i = 0; i < files.size(); i++)
	{
		if (files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

	return files.size();
}

struct PrepImageItem
{
	int id;
	bool isQueud;
	ImageAndExposure *pt;

	inline PrepImageItem(int _id)
	{
		id = _id;
		isQueud = false;
		pt = 0;
	}

	inline void release()
	{
		if (pt != 0)
			delete pt;
		pt = 0;
	}
};

class ImageFolderReader
{
public:
	ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		this->path = path;
		this->calibfile = calibFile;

#if HAS_ZIPLIB
		ziparchive = 0;
		databuffer = 0;
#endif

		isZipped = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");

		if (isZipped)
		{
#if HAS_ZIPLIB
			int ziperror = 0;
			ziparchive = zip_open(path.c_str(), ZIP_RDONLY, &ziperror);
			if (ziperror != 0)
			{
				printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
				exit(1);
			}

			files.clear();
			int numEntries = zip_get_num_entries(ziparchive, 0);
			for (int k = 0; k < numEntries; k++)
			{
				const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if (nstr == "." || nstr == "..")
					continue;
				files.push_back(name);
			}

			printf("got %d entries and %d files!\n", numEntries, (int)files.size());
			std::sort(files.begin(), files.end());
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
		else
			getdir(path, files);

		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width = undistort->getSize()[0];
		height = undistort->getSize()[1];

		// load timestamps if possible.
		loadTimestamps();
		printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());
	}
	~ImageFolderReader()
	{
#if HAS_ZIPLIB
		if (ziparchive != 0)
			zip_close(ziparchive);
		if (databuffer != 0)
			delete databuffer;
#endif

		delete undistort;
	};

	Eigen::VectorXf getOriginalCalib()
	{
		return undistort->getOriginalParameter().cast<float>();
	}
	Eigen::Vector2i getOriginalDimensions()
	{
		return undistort->getOriginalSize();
	}

	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		getCalibMono(K, w_out, h_out);
		setGlobalCalib(w_out, h_out, K);
	}

	int getNumImages()
	{
		return files.size();
	}

	double getTimestamp(int id)
	{
		if (timestamps.size() == 0)
			return id * 0.1f;
		if (id >= (int)timestamps.size())
			return 0;
		if (id < 0)
			return 0;
		return timestamps[id];
	}

	void prepImage(int id, bool as8U = false)
	{
	}

	MinimalImageB *getImageRaw(int id)
	{
		return getImageRaw_internal(id, 0);
	}

	ImageAndExposure *getImage(int id, bool forceLoadDirectly = false)
	{
		return getImage_internal(id, 0);
	}

	inline float *getPhotometricGamma()
	{
		if (undistort == 0 || undistort->photometricUndist == 0)
			return 0;
		return undistort->photometricUndist->getG();
	}

	// undistorter. [0] always exists, [1-2] only when MT is enabled.
	Undistort *undistort;

private:
	MinimalImageB *getImageRaw_internal(int id, int unused)
	{
		if (!isZipped)
		{
			// CHANGE FOR ZIP FILE
			return IOWrap::readImageBW_8U(files[id]);
		}
		else
		{
#if HAS_ZIPLIB
			if (databuffer == 0)
				databuffer = new char[widthOrg * heightOrg * 6 + 10000];
			zip_file_t *fle = zip_fopen(ziparchive, files[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 6 + 10000);

			if (readbytes > (long)widthOrg * heightOrg * 6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes, (long)widthOrg * heightOrg * 6 + 10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg * heightOrg * 30];
				fle = zip_fopen(ziparchive, files[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 30 + 10000);

				if (readbytes > (long)widthOrg * heightOrg * 30)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes, (long)widthOrg * heightOrg * 30 + 10000);
					exit(1);
				}
			}

			return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
	}

	ImageAndExposure *getImage_internal(int id, int unused)
	{
		MinimalImageB *minimg = getImageRaw_internal(id, 0);
		ImageAndExposure *ret2 = undistort->undistort<unsigned char>(
			minimg,
			(exposures.size() == 0 ? 1.0f : exposures[id]),
			(timestamps.size() == 0 ? 0.0 : timestamps[id]));
		delete minimg;
		return ret2;
	}

	inline void loadTimestamps()
	{
		std::ifstream tr;
		std::string timesFile = path.substr(0, path.find_last_of('/')) + "/times.txt";
		tr.open(timesFile.c_str());
		while (!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int id;
			double stamp;
			float exposure = 0;

			if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if (2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
		}
		tr.close();

		// check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int)exposures.size() == (int)getNumImages());
		for (int i = 0; i < (int)exposures.size(); i++)
		{
			if (exposures[i] == 0)
			{
				// fix!
				float sum = 0, num = 0;
				if (i > 0 && exposures[i - 1] > 0)
				{
					sum += exposures[i - 1];
					num++;
				}
				if (i + 1 < (int)exposures.size() && exposures[i + 1] > 0)
				{
					sum += exposures[i + 1];
					num++;
				}

				if (num > 0)
					exposures[i] = sum / num;
			}

			if (exposures[i] == 0)
				exposuresGood = false;
		}

		if ((int)getNumImages() != (int)timestamps.size())
		{
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if ((int)getNumImages() != (int)exposures.size() || !exposuresGood)
		{
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
	}

	std::vector<ImageAndExposure *> preloadedImages;
	std::vector<std::string> files;
	std::vector<double> timestamps;
	std::vector<float> exposures;

	int width, height;
	int widthOrg, heightOrg;

	std::string path;
	std::string calibfile;

	bool isZipped;

#if HAS_ZIPLIB
	zip_t *ziparchive;
	char *databuffer;
#endif
};

class readerEuRoC
{
public:
	// 构造函数
	readerEuRoC(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		// 记录图片数据位置和标定参数位置
		this->path = path;
		this->calibfile = calibFile;
		// 获取所有图像名称的string + 时间戳 + 曝光
		loadTimestamps();
		// 拿到标定参数 TODO
		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);
		// 具体的标定数据记录
		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width = undistort->getSize()[0];
		height = undistort->getSize()[1];
		printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());
		{ // 临时加的一段
			std::string path = "/home/lyc/slam/dataSet/data_odometry_poses/dataset/poses/00.txt";
			std::ifstream file;
			std::string s;
			file.open(path.c_str());
			//std::getline(file, s);
			while (!file.eof())
			{
				std::getline(file, s);
				if (!s.empty())
				{
					std::stringstream ss(s);
					std::string data;
					Eigen::Vector3f t;
					//std::getline(ss, data, ',');
					std::getline(ss, data, ' ');
					std::stringstream(data) >> t[0];
					std::getline(ss, data, ' ');
					std::stringstream(data) >> t[1];
					std::getline(ss, data, ' ');
					std::stringstream(data) >> t[2];
					trace.push_back(t);
				}
			}
		}
	}
	// 设置全局标定参数
	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		// 获取设置参数
		getCalibMono(K, w_out, h_out);
		// 得到内部全局
		setGlobalCalib(w_out, h_out, K);
	}
	inline float *getPhotometricGamma()
	{
		if (undistort == 0 || undistort->photometricUndist == 0)
			return 0;
		return undistort->photometricUndist->getG();
	}
	int getNumImages()
	{
		return files.size();
	}
	double getTimestamp(int id)
	{
		if (timestamps.size() == 0)
			return id * 0.1f;
		if (id >= (int)timestamps.size())
			return 0;
		if (id < 0)
			return 0;
		return timestamps[id];
	}
	ImageAndExposure *getImage(int id, bool forceLoadDirectly = false)
	{
		return getImage_internal(id, 0);
	}
	Undistort *undistort;
	std::vector<Eigen::Vector3f> trace;

private:
	std::string path;
	std::string calibfile;
	std::vector<std::string> files;
	std::vector<double> timestamps;
	std::vector<float> exposures;
	int width, height;
	int widthOrg, heightOrg;
	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}
	void loadTimestamps()
	{
		std::ifstream fTimes;
		std::string s;
		std::string strPathTimes = path + "/data.csv";
		fTimes.open(strPathTimes.c_str());

		std::getline(fTimes, s);
		while (!fTimes.eof())
		{
			std::getline(fTimes, s);
			if (!s.empty())
			{
				std::stringstream ss(s);
				std::string time;
				std::getline(ss, time, ',');
				files.push_back(path + "/data/" + time + ".png");
				double t;
				std::stringstream(time) >> t;
				timestamps.push_back(t / 1e9);
				float e;
				std::getline(ss, time, ',');
				std::getline(ss, time, ',');
				std::stringstream(time) >> e;
				exposures.push_back(e / 1e6);
			}
		}
	}
	ImageAndExposure *getImage_internal(int id, int unused)
	{
		MinimalImageB *minimg = IOWrap::readImageBW_8U(files[id]);
		// cv::Mat test(cv::Size(minimg->w, minimg->h), cv::IMREAD_GRAYSCALE);
		// memcpy(test.data,minimg->data,minimg->w * minimg->h);
		// cv::imshow("test", test);
		// cv::waitKey(0);
		ImageAndExposure *ret2 = undistort->undistort<unsigned char>(
			minimg,
			(exposures.size() == 0 ? 1.0f : exposures[id]),
			(timestamps.size() == 0 ? 0.0 : timestamps[id]));
		delete minimg;
		return ret2;
	}
};

// 读取kitti数据集
class readerKitti
{
public:
	// 构造函数
	readerKitti(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		// 记录图片数据位置和标定参数位置
		this->path = path;
		this->calibfile = calibFile;
		// 获取所有图像名称的string + 时间戳 + 曝光
		getdir(path + "/image_0", files);
		loadTimestamps();
		// 拿到标定参数 TODO
		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);
		// 具体的标定数据记录
		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width = undistort->getSize()[0];
		height = undistort->getSize()[1];
		printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());
		{ // 临时加的一段
			std::string path = "/home/lyc/slam/dataSet/data_odometry_poses/dataset/poses/03.txt";
			FILE *fp = fopen(path.c_str(), "r");
			float m[3][4];
			//std::getline(file, s);
			while (!feof(fp))
			{

				if (fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f",
						   &m[0][0], &m[0][1], &m[0][2], &m[0][3],
						   &m[1][0], &m[1][1], &m[1][2], &m[1][3],
						   &m[2][0], &m[2][1], &m[2][2], &m[2][3]) == 12)
				{
					Eigen::Vector3f t;
					t[0] = m[0][3];
					t[1] = m[1][3];
					t[2] = m[2][3];
					trace.push_back(t);
				}
			}
		}
	}
	// 设置全局标定参数
	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		// 获取设置参数
		getCalibMono(K, w_out, h_out);
		// 得到内部全局
		setGlobalCalib(w_out, h_out, K);
	}
	inline float *getPhotometricGamma()
	{
		if (undistort == 0 || undistort->photometricUndist == 0)
			return 0;
		return undistort->photometricUndist->getG();
	}
	int getNumImages()
	{
		return files.size();
	}
	double getTimestamp(int id)
	{
		if (timestamps.size() == 0)
			return id * 0.1f;
		if (id >= (int)timestamps.size())
			return 0;
		if (id < 0)
			return 0;
		return timestamps[id];
	}
	ImageAndExposure *getImage(int id, bool forceLoadDirectly = false)
	{
		return getImage_internal(id, 0);
	}
	Undistort *undistort;
	std::vector<Eigen::Vector3f> trace;

private:
	std::string path;
	std::string calibfile;
	std::vector<std::string> files;
	std::vector<double> timestamps;
	std::vector<float> exposures;
	int width, height;
	int widthOrg, heightOrg;
	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

	void loadTimestamps()
	{
		std::ifstream fTimes;
		std::string s;
		std::string strPathTimes = path + "/times.txt";
		fTimes.open(strPathTimes.c_str());

		while (!fTimes.eof())
		{
			std::getline(fTimes, s);
			double t;
			std::stringstream(s) >> t;
			timestamps.push_back(t);
		}
	}
	ImageAndExposure *getImage_internal(int id, int unused)
	{
		MinimalImageB *minimg = IOWrap::readImageBW_8U(files[id]);
		// cv::Mat test(cv::Size(minimg->w, minimg->h), cv::IMREAD_GRAYSCALE);
		// memcpy(test.data,minimg->data,minimg->w * minimg->h);
		// cv::imshow("test", test);
		// cv::waitKey(0);
		ImageAndExposure *ret2 = undistort->undistort<unsigned char>(
			minimg,
			(exposures.size() == 0 ? 1.0f : exposures[id]),
			(timestamps.size() == 0 ? 0.0 : timestamps[id]));
		delete minimg;
		return ret2;
	}
};
