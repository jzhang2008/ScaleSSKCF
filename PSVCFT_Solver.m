function [conj_w,b_mat]=PSVCFT_Solver(params, denominator, xlf)
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
if nargin<6,Iter=500; end
if nargin<7,thre=1e-6;end
gamma    = 0.01;
beta     = 0.01;
beta_max = 10;
alpha    = 1.1;
[row,col,fdim,K]=size(xlf);
lambda =1/params.C;
% initial the related parameters
theta_pre = zeros(row,col,fdim,K);
w_pre     = zeros(row,col,fdim,K);
v_pre     = zeros(row,col,fdim,K);
% conj_w    = zeros(row,col,fdim,K,'single');
 % initial b ,y
 part_y  = params.part_y;
%  part_yf = params.part_yf;
 b       = mean(part_y(:));
 mpart_y = repmat(part_y,[1,1,K]);
 b_mat   = b*ones(row,col,K);
 numerator_pre = zeros(row,col,fdim,K,'single');
 
%  y  = params.yy;
%  yf = params.yf;
%  b  = mean(y(:));
%  % Initialize numerator
%  numerator_pre = bsxfun(@times, yf, conj(xlf));
%  % update matrix e and numerator    
%  [ row, col ] = find( y == 0 );
%  w_pre =conj(fft2(params.w));
conj_w  = conj(fft2(w_pre));
for i=1:Iter
    w_init  = real(sum(w_pre-v_pre-theta_pre/beta,4))/K;
    ww_init = repmat(w_init,[1 1 1 K]);
    v       = shrinkage(real(w_pre+theta_pre/beta-ww_init),gamma/beta);
    para_f  = real(ifft2(sum(xlf.*conj_w,3)));
    tpara_f = permute(para_f,[1,2,4,3]);
    para_e  = mpart_y.*(tpara_f+b_mat)-1;
    para_e  = max(0,para_e);
    % update p
    q       = mpart_y + mpart_y.*para_e;
    b       = mean(reshape(q,[row*col,K]));
    b_mat   = repmat(b,[row*col,1]);
    b_mat   = reshape(b_mat,[row,col,K]);
    p       = q-b_mat;
    pf      = fft2(p);
    hf_den  = denominator +lambda+lambda*beta/2;
    for j= 1:K
        numerator_pre(:,:,:,j)=bsxfun(@times,pf(:,:,j),conj(xlf(:,:,:,j)));
    end
    hf_num  = numerator_pre -lambda/2*fft2(theta_pre)+beta/2*lambda*(fft2(v)+fft2(ww_init));
    for j= 1:K
        conj_w(:,:,:,j) = bsxfun(@times,1./hf_den(:,:,j),hf_num(:,:,:,j)); 
    end
    w=ifft2(conj(conj_w));
    % update theta
    theta_pre = theta_pre+beta*(w-v-ww_init);
    beta      = min(beta_max,alpha*beta);
    % stopping criteria
    r1 = w(:);
    r2 = w_pre(:);
    obj  = real(sum(sum(sum(sum(w.^2))))+params.C*sum(sum(sum((tpara_f+b_mat-q).^2))));
    if i==1
        flag = 1;
        flag_obj  = 1;
    else
        flag = norm(r1(:)-r2(:))^2/norm(r1(:))^2;
        flag_obj  = abs(obj_old-obj);
    end
    
    if flag<thre ||i==Iter || flag_obj<0.1
        break;
    else
        w_pre = w;
        v_pre = v;
        obj_old =obj;
    end
    
%         f = real(ifft2(sum(numerator_pre .* xf, 3) ./ (denominator + lambda))) + b;
%         para_e = y .* f - 1;
%         
%         para_e(para_e<0)=0; 
%         for k = 1 : size(row)
%             para_e(row(k),col(k)) = max( 0, f(row(k),col(k)) );
%         end
% 
%         % update p
%         q = y + y .* para_e;
%         b = mean( q( : ) );
%         p = q - b;
%         numerator = bsxfun(@times, fft2(p), conj(xf));
% 
%        %% Í£Ö¹Ìõ¼þÅÐ¶Ï
% 		t1 = real( numerator ); t2 = real( numerator_pre );   
%         if i == 1
%             flag = 1;
%         else
%             flag = norm( ( t1(:) - t2(:) ) )^2 / norm( t1(:) )^2;
%         end
%         if ( flag < thre || i==Iter )
%             model_numerator = numerator;
%             break;
%         else
%             numerator_pre = numerator;
%         end;
%     d=y.*(ifft2(xlf.*w_pre)+b)-1; %
%     e=max(0,d);
%     q=y+y.*e;
%     b=mean2(q);
% %     b=mean2(q-ifft2(conj(xlf).*w0));
%     p=q-b;
%     pf=fft2(p);
%     hf_den =conj(xlf).*xlf+lambda;
%     hf_num =bsxfun(@times,pf,conj(xlf));
%     w      = hf_num./hf_den;
%     r1     = real(w);
%     r2     = real(w_pre);
%     if  i==1
%        flag=1;
%     else
%        flag=norm(r1(:)-r2(:))^2/norm(r1(:))^2;
%     end
%     obj    =sum(sum(ifft2(w).^2))+1/lambda*sum(sum((y.*(ifft2(xlf.*w)+b)-1-e).^2)); %
%     if flag<thre ||i==Iter
%         break;
%     else
%      w_re=w;
%     end
end
end
function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
%     zz=max(0,abs(x)-kappa).*sign(x);
end 



