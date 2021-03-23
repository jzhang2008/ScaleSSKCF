function [w,b]=SVKCFT_Solver(params,  kf)
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
lambda   = 1/params.C;
beta_est = params.beta/params.C;
w0       = params.ww;
% epsilon  =1.5;
 % initial b
 y  = params.part_y;
%  yf = params.yf;
 b  = mean(y(:));
 % Initialize numerator
%  numerator_pre = bsxfun(@times, yf, conj(xlf));
 w             = zeros(size(kf));
 % update matrix e and numerator    
 [ row, col ] = find( y == 0 );
%  w_pre =yf ./ (kf + lambda/(1+lambda*epsilon));  %初始化alphaf;
 w_pre        = conj(fft2(w));
for i=1:Iter
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
%        %% 停止条件判断
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
    f  =real(ifft2(kf.*w_pre))+b;
    para_e=y.*f-1; %
    para_e(para_e<0)=0;
    for k = 1 : size(row)
            para_e(row(k),col(k)) = max( 0, f(row(k),col(k)) );
    end
    q=y+y.*para_e;
    b=mean2(q);
%     b=mean2(q-ifft2(conj(xlf).*w0));
    p=q-b;
    pf=fft2(p);
    hf_num =pf+beta_est*w0;
    w      = bsxfun(@times,hf_num,1./(kf+lambda+beta_est));
    r1     = real(w);
    r2     = real(w_pre);
    % obj    =sum(sum(sum(real(ifft2(conj(w))).^2)))+1/lambda*sum(sum((y.*(ifft2(sum(xlf.*w,3))+b)-1-para_e).^2)); 
    if  i==1
       flag     =1;
%        flag_obj =1;
    else
       flag     = norm(r1(:)-r2(:))^2/norm(r1(:))^2;
%        flag_obj = abs(obj-obj_old);
    end
    
    if flag<thre ||i==Iter 
        break;
    else
     w_pre    =  w;
%      obj_old = obj;
    end
end
end



