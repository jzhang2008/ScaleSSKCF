function [WW]=SparseCorrelationFilterSolver(hf_den,hf_num,lambda,rol,Iter)
%objectfunction
% min_w 1/2||Xw-y||^2+lambda||w||_1
%usage
%[]=SparseCorrelationFilterSolver(hf_den,hf_num,lambda,rol)
%INPUTS
%
%
%output:
%
%
%copyrighted by jzhang
% email:jizhangjian@sxu.edu.cn
if(nargin<5),Iter=50;end
if(nargin<4),rol=10;end
if(nargin<3),lambda=0.01;end
[row,col,dim]=size(hf_num);
beta    =zeros(row,col,dim);
beta_old=zeros(row,col,dim);
t       =zeros(row,col,dim);
t_old   =zeros(row,col,dim);
res =zeros(1,Iter);
RELTOL    = 0.5*2*1e-2;
alpha_old =1;
lambda  = repmat(lambda,[1 1 dim]);
for k=1:Iter
    f_beta=fft2(beta);
    f_t   =fft2(t);
    f_w   =bsxfun(@times,(hf_num-f_beta+rol*f_t),1./(hf_den+rol));
    t_new =shrinkage(real(ifft2(f_w)+ifft2(f_beta)/rol),lambda/rol);
    beta  =beta+rol*(ifft2(f_w)-t_new);
%     alpha =(1+sqrt(1+4*alpha_old^2))/2;
%     t_new =t_new+(alpha_old-1)/alpha*(t_new-t_old);
%     beta  =beta +(alpha_old-1)/alpha*(beta-beta_old);
    if max(max(sum( abs(ifft2(f_w)-t_new),2)))<RELTOL||k>=Iter  %&& max(max(sum( abs(ifft2(f_w-f_w_old)),2)))<RELTOL)
        break;
    end
    res(k)=max(max(sum( abs(ifft2(f_w)-t_new),2)));
    t        =t_new;
%     t_old    =t_new;
%     beta_old =beta;
%     alpha_old=alpha;
    f_w_old=f_w;
end
WW=fft2(t_new);
end
function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
%     zz=max(0,abs(x)-kappa).*sign(x);
end

