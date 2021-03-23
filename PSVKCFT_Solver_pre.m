function [conj_w,b_mat]=PSVKCFT_Solver_pre(params, kf)
% model_numerator
%solving the support vector correlation filter tracker
% input parameters:
%                 xlf: feature vector of object
%                  y : class labels
%                 w0 : initialized filter coefficients
%                 b0 : initialized bias value
%                 C  : regularization parameter
% output:
%                 w  : final filter coefficients
%                 b  : final bias value
% copyrighted by jzhang
% date:2017/4/21
if nargin<6,Iter=10; end
if nargin<7,thre=2e-3;end
% gamma    = 0.01;
beta       = params.beta; %0.01;%
% beta_max = 10;
% alpha    = 1.1;
[row,col,K] = size(kf);
lambda =1/params.C;
w_pre     = params.w_pre;
pw        = params.ppw;
% initial the related parameters
% theta_pre = zeros(row,col,fdim,K);
% w_pre     = zeros(row,col,fdim,K);
% v_pre     = zeros(row,col,fdim,K);
conj_w    = zeros(row,col,K,'single');
ppw       = repmat(pw,[row*col,1]);
ppw       = reshape(ppw,[row,col,K]);
 % initial b ,y
 part_y  = params.part_y;
%  part_yf = params.part_yf;
 b       = mean(part_y(:));
 mpart_y = repmat(part_y,[1,1,K]);
 b_mat   = b*ones(row,col,K);
%  numerator_pre = zeros(row,col,K,'single');
%  ww_init_pre   = zeros(row,col,K,'single');
%  y  = params.yy;
%  yf = params.yf;
%  b  = mean(y(:));
%  % Initialize numerator
%  numerator_pre = bsxfun(@times, yf, conj(xlf));
%  % update matrix e and numerator    
%  [ row, col ] = find( y == 0 );
%  w_pre =conj(fft2(params.w));
%  conj_w  = params.w0;
for i=1:Iter
%     w_init  = real(sum(w_pre-v_pre-theta_pre/beta,4))/K;
%     w_init  = real(sum(w_pre.*ppw,3))/sum(pw);
    w_init  = real(sum(w_pre,3))/K;
    ww_init = repmat(w_init,[1 1 K]);
%     v       = shrinkage(real(w_pre+theta_pre/beta-ww_init),gamma/beta);
    para_f  = real(ifft2(kf.*conj_w))+b_mat;
    para_e  = mpart_y.*para_f-1;
    para_e(para_e<0)=0;
    % update p
    q       = mpart_y + mpart_y.*para_e;
    b       = mean(reshape(q,[row*col,K]));
    b_mat   = repmat(b,[row*col,1]);
    b_mat   = reshape(b_mat,[row,col,K]);
    p       = q-b_mat;
    pf      = fft2(p);
%     pw      = repmat(params.ppw,[numel(kf)/K,1]);
%     pw      = reshape(pw,size(kf));
    hf_den  = kf +lambda+beta*ppw+params.eta;%
    hf_num  = pf + beta*ppw.*fft2(ww_init)+params.eta*conj(params.w0); %
    conj_w  = hf_num./hf_den;
%     for j= 1:K
%         conj_w(:,:,:,j) = bsxfun(@times,1./hf_den(:,:,j),hf_num(:,:,:,j)); 
%     end
    w=ifft2(conj(conj_w));
    % update theta
%     theta_pre = theta_pre+beta*(w-v-ww_init);
%     beta      = min(beta_max,alpha*beta);
    % stopping criteria
    r1 = w(:);
    r2 = w_pre(:);
%     obj  = real(sum(sum(sum(sum(w.^2))))+params.C*sum(sum(sum((tpara_f+b_mat-q).^2))));
    if i==1
        flag = 1;
%         flag_obj  = 1;
    else
        flag    = norm(r1(:)-r2(:))^2/norm(r1(:))^2;
%         flag  = norm(para_f(:)-f_old(:),2)/norm(para_f(:),2);
%         flag_obj  = abs(obj_old-obj);
    end
    
    if flag<thre ||i==Iter %|| flag_obj<0.1
        break;
    else
        w_pre = w;
        f_old = para_f;
%         v_pre = v;
%         obj_old =obj;
    end
end
end
function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
%     zz=max(0,abs(x)-kappa).*sign(x);
end 



