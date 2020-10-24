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


 
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{


PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
	instanceCounter++;
	host = rawPoint->host;
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;

	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;

	efPoint=0;


}


void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}


void FrameHessian::setStateZero(const Vec10 &state_zero)
{
	assert(state_zero.head<6>().squaredNorm() < 1e-20);

	this->state_zero = state_zero;


	for(int i=0;i<6;i++)
	{
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3::exp(eps);
		SE3 EepsM = Sophus::SE3::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	// scale change
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);


	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};



void FrameHessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL
	for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
	for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
	for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}

// 在传入一帧图像的初始调用
// color 为图像
// HCalib 为标定的参数
void FrameHessian::makeImages(float* color, CalibHessian* HCalib)
{
	// 遍历使用的金字塔层数
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		// 创建金字塔大小的图片 图片是三层的 像素 / dx / dy
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
		// 创建金字塔大小的图片，保存梯度平方信息，用于筛选得到点
		absSquaredGrad[i] = new float[wG[i]*hG[i]];
	}
	// dI 为原始图片，指向金字塔最顶层的图片
	dI = dIp[0];

	// 最顶层，和原始图像保持一致
	// make d0
	int w=wG[0];
	int h=hG[0];
	for(int i=0;i<w*h;i++)
		dI[i][0] = color[i];
	// 遍历金字塔所有层数
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		// 得到当前金字塔图像大小
		int wl = wG[lvl], hl = hG[lvl];
		// 当前指针指向对应的金字塔图片
		Eigen::Vector3f* dI_l = dIp[lvl];
		// 梯度指针指向金字塔梯度图片
		float* dabs_l = absSquaredGrad[lvl];
		if(lvl>0)
		{// 从第二层开始，最顶层已经赋值了
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f* dI_lm = dIp[lvlm1];
			// 向下采样得到金字塔的图像信息，TODO这里应该可以进行一定的加速处理
			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}
		// 从第二行开始计算到倒数第二行
		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			// 计算梯度
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);
			// 避免无穷大的情况
			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;
			// 记录一下梯度
			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;
			// 得到绝对值
			dabs_l[idx] = dx*dx+dy*dy;

			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
				// 计算梯度，注意这里要考虑相机参数对梯度的影响
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
				dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
			}
		}
	}
}

void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
	this->host = host;
	this->target = target;

	SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
	PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();



	SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	PRE_tTll = (leftToLeft.translation()).cast<float>();
	distanceLL = leftToLeft.translation().norm();


	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;
	PRE_KRKiTll = K * PRE_RTll * K.inverse();
	PRE_RKiTll = PRE_RTll * K.inverse();
	PRE_KtTll = K * PRE_tTll;


	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	PRE_b0_mode = host->aff_g2l_0().b;
}

}

