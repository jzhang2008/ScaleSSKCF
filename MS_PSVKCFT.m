function [positions, fps] = MS_PSVKCFT(params)

% [positions, fps] = MS_PSVCFT(params)

% parameters
% search_area_scale = params.search_area_scale;
% padding = params.padding;                         	%extra area surrounding the target
output_sigma_factor = params.output_sigma_factor;	%spatial bandwidth (proportional to target)
lambda = params.lambda;
% K_num                = params.K;
learning_rate = params.learning_rate;
nScales = params.number_of_scales;
scale_step = params.scale_step;

refinement_iterations = params.refinement_iterations;
filter_max_area = params.filter_max_area;
interpolate_response = params.interpolate_response;
% num_GS_iter = params.num_GS_iter;
features = params.t_features;

img_files = params.img_files;
pos = floor(params.init_pos);
target_sz = floor(params.wsize);

debug = params.debug;
visualization = params.visualization || debug;
% visualization = params.visualization;

num_frames = numel(img_files);

init_target_sz = target_sz;

% constructed spatial layout model
Aspect_ratio             = init_target_sz(1)/init_target_sz(2);            % Aspect ratio
params.pflag     =0;
if params.objRatio<0.04 && params.objRatio>0.02 && Aspect_ratio>0.6 && Aspect_ratio<=1.6
    params.pflag =1;
    params.search_area_scale = 3.5;
end
search_area_scale = params.search_area_scale;
[K_num,part_sz,part_pos] = constructed_layout(init_target_sz,Aspect_ratio,pos,params);
%set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;
search_area = prod(init_target_sz / featureRatio * search_area_scale);
% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        featureRatio = params.t_global.cell_size;
        search_area  = prod(init_target_sz / featureRatio * search_area_scale);
%         flag_area    = 1;
%         K_num        = 4;
%         part_layout  = [2,2];
%         part_sz      = ceil(init_target_sz./part_layout);
%         part_search_area=prod(part_sz/ featureRatio * search_area_scale);
    end
end

% the relative parameters of part
% part_sz          = ceil(init_target_sz./part_layout);
% part_search_area = prod(part_sz/ featureRatio * search_area_scale);

global_feat_params = params.t_global;
if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end
% target size at the initial scale
base_target_sz     = target_sz/currentScaleFactor;
base_part_sz       = part_sz/currentScaleFactor;
appearance_part_sz = round(base_part_sz);
% part_sz        = base_part_sz*currentScaleFactor;
% constructed the initialized pos of each part
% if K_num==6
%     idx_x  =[-1 0 1]'*[1 1];
%     idx_y  =[1 1 1]'*[-0.5 0.5];
%     pos_vec=floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
% else
%     idx_x  =[-0.5 0.5]'*[1 1];
%     idx_y  =[1 1 ]'*[-0.5 0.5];
%     pos_vec=floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
% end
% part_pos =ones(K_num,1)*pos+pos_vec;
%window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz  = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
        psz = repmat(sqrt(prod(base_part_sz * search_area_scale)), 1, 2);   % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end
% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
use_sz = floor(sz/featureRatio);
psz = round(psz / featureRatio) * featureRatio;
use_psz = floor(psz/featureRatio);
% window size, taking padding into account
% sz = floor(base_target_sz * (1 + padding));

% desired translation filter output (gaussian shaped), bandwidth
% proportional to target size
% construct the label function
% the label of parts_region
part_sigma   = sqrt(prod(floor(base_part_sz/featureRatio))) * params.part_sigma_factor;
part_labels  = gaussian_shaped_labels(part_sigma, use_psz);
part_y       = binary_labeling(part_labels,params);
params.part_y=part_y;
params.part_yf=fft2(part_y);

if interpolate_response == 1
    interp_psz = use_psz * featureRatio;
else
    interp_psz = use_psz;
end

output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
labels   = gaussian_shaped_labels(output_sigma, use_sz);
% yy       = binary_labeling(labels,params);
params.yy =labels;
params.yf=fft2(labels);


if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales
% scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
% ss = (1:nScales) - ceil(nScales/2);
% ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
% ysf = single(fft(ys));

% store pre-computed translation filter cosine window
cos_window   =  single(hann(use_sz(1)) * hann(use_sz(2))');
part_cos_win =  single(hann(use_psz(1)) * hann(use_psz(2))');
% create video interface
% if  visualization,  
%     update_visualization = show_video(img_files, 0);
% end

% the search area size
% support_sz = prod(use_sz);
% Calculate feature dimension
im = imread(img_files{1});
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end
if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end
% store pre-computed scale filter cosine window
% if mod(nScales,2) == 0
%     scale_window = single(hann(nScales+1));
%     scale_window = scale_window(2:end);
% else
%     scale_window = single(hann(nScales));
% end;
% if params.use_reg_window
%     % create weight window
%     % normalization factor
%     reg_scale = 0.5 * base_target_sz/featureRatio;
%     % construct grid
%     wrg = -(use_sz(1)-1)/2:(use_sz(1)-1)/2;
%     wcg = -(use_sz(2)-1)/2:(use_sz(2)-1)/2;
%     [wrs, wcs] = ndgrid(wrg, wcg);
%     % construct the regukarization window
%     reg_window = exp(-(abs(wrs/reg_scale(1)).^params.reg_window_power + abs(wcs/reg_scale(2)).^params.reg_window_power)/16);
% end
% scale factors
if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    
    scaleFactors = scale_step .^ scale_exp;
    
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% compute the resize dimensions used for feature extraction in the scale
% estimation
% scale_model_factor = 1;
% if prod(init_target_sz) > scale_model_max_area
%     scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
% end
% scale_model_sz = floor(init_target_sz * scale_model_factor);
if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
%     ky = -floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2);
%     kx = -floor((use_sz(2) - 1)/2): ceil((use_sz(2) - 1)/2);
%     kx = kx';
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end
% currentScaleFactor = 1;

% to calculate precision
positions = zeros(numel(img_files), 4);

% to calculate FPS
time = 0;

% find maximum and minimum scales
% im = imread([video_path img_files{1}]);
% min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
% max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));

global_pixels            = zeros(psz(1), psz(2), size(im,3),'uint8');
parts_pixels             = zeros(psz(1), psz(2), size(im,3), K_num, 'uint8');
multires_pixel_template  = zeros(psz(1), psz(2), size(im,3), K_num, 'uint8');
appearance_parts_pixels  = zeros(appearance_part_sz(1),appearance_part_sz(2),size(im,3),K_num);
kf                       = zeros(use_psz(1), use_psz(2), K_num, 'single'); 
translation_vec_old      = [0 0];
sim_part                 = ones(1,K_num);
for frame = 1:num_frames,
    %load image
    im = imread(img_files{frame});
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    tic;
%     if frame==184
%         system('pause');
%     end
    if frame > 1
        old_pos = inf(size(pos));
        iter = 1;
%         xt   = zeros(use_psz(1), use_psz(2), feature_dim, nScales, K_num, 'single');
        %translation search
        while iter <= refinement_iterations && any(old_pos ~= pos)
%             for part_ind = 1:K_num
%                  % Get multi-resolution image
%                  for scale_ind = 1:nScales
%                      multires_pixel_template(:,:,:,scale_ind,part_ind) = ...
%                          get_pixels(im, part_pos(part_ind,:), round(psz*currentScaleFactor*scaleFactors(scale_ind)), psz);
%                  end
%                 xx                  = get_features(multires_pixel_template(:,:,:,:,part_ind),features,global_feat_params);
%                 xt(:,:,:,:,part_ind) = bsxfun(@times,xx,part_cos_win);
%             end
             for part_ind = 1:K_num
                 % Get multi-resolution image
                     parts_pixels(:,:,:,part_ind) = ...
                         get_pixels(im, part_pos(part_ind,:), round(psz*currentScaleFactor), psz);
             end
%                 xx         = get_features(multires_pixel_template,features,global_feat_params);
%                 xt         = bsxfun(@times,xx,part_cos_win);
            xt  = bsxfun(@times,get_features(parts_pixels,features,global_feat_params),part_cos_win);
            % calculate the correlation response of the translation filter
            xtf       = fft2(xt);
            for part_ind =1:K_num
                kf(:,:,part_ind)    = gaussian_correlation(xtf(:,:,:,part_ind), xlf(:,:,:,part_ind), params.kernel.sigma);
            end
            responsef = hf_mode_alpha.*kf;%+hf_bias;
%             xtf = permute(xtf,[1 2 3 5 4]);
%             xtf = cellfun(@(x) fft2(x),xt,'uniformoutput', false);
%             hf_m=cell2mat(hf_mode_alpha);
%             xtf_m=cell2mat(xtf);
%             responsef = permute(sum(bsxfun(@times,hf_m, xtf_m), 3),[1 2 4 3]);
%             responsef = cellfun(@(hf,x) permute(sum(bsxfun(@times,hf, x), 3),[1 2 4 3]),hf_mode_alpha,xtf,'uniformoutput', false);
%             responsef = sum(cat(4,responsef{:}),4);
%             responsef = permute(sum(bsxfun(@times,hf_mode_alpha, xtf), 3),[1 2 4 5 3]);
             % if we undersampled features, we want to interpolate the
            % response so it has the same size as the image patch
            if interpolate_response == 2
                % use dynamic interp size
                interp_sz = floor(size(part_y) * featureRatio * currentScaleFactor);
            end
            responsef_padded = resizeDFT2(responsef, interp_psz);
            
            % response
            response = ifft2(responsef_padded, 'symmetric');
            % find each maximum
             resp_max   = zeros(1,K_num);
            resp_mean  = zeros(1,K_num);
            resp_std   = zeros(1,K_num);
            row        = zeros(1,K_num);
            col        = zeros(1,K_num);    
            for part_ind =1:K_num
                   resp_max(part_ind)     = max(max(real(response(:,:,part_ind))));
                  resp_mean(part_ind)    = mean2(real(response(:,:,part_ind)));
                  resp_std(part_ind)     = std2(real(response(:,:,part_ind)));
                 [row(part_ind), col(part_ind), sind] =ind2sub(size(response(:,:,part_ind)),find(response(:,:,part_ind)==resp_max(part_ind),1));
            end
%             disp_row = mod(row- 1 + floor((interp_psz(1)-1)/2), interp_psz(1)) - floor((interp_psz(1)-1)/2);
%             disp_col = mod(col - 1 + floor((interp_psz(2)-1)/2), interp_psz(2)) - floor((interp_psz(2)-1)/2);
            %--------------------------------------------------------------------------------------------------------
            % calculate PSR value
%             cm_psr   = zeros(1,K_num);
%             for part_ind =1:K_num
%                 cm_psr(part_ind) = PSR (response(:,:,part_ind),params.PSR_range);
%             end
            PSR                        = (resp_max-resp_mean)./resp_std;
            % Judge whether is occluded
            flag_occ1      = PSR>5.5;                                         % if PSR< 5, we consider that it is not reliable
            part_disp_row              = mod(row - 1 + floor((interp_psz(1)-1)/2), interp_psz(1)) - floor((interp_psz(1)-1)/2);
            part_disp_col              = mod(col - 1 + floor((interp_psz(2)-1)/2), interp_psz(2)) - floor((interp_psz(2)-1)/2);
            part_vec                   = round([part_disp_row; part_disp_col]*featureRatio * currentScaleFactor)';
%             % calculate SCCM value
%             for part_ind =1:K_num
%                 response_old(:,:,part_ind) = circshift(response_old(:,:,part_ind),part_vec(part_ind,:));
%             end
%             SCCM                       = sqrt(sum((reshape(response,[use_psz(1)*use_psz(2), K_num])-reshape(response_old,[use_psz(1)*use_psz(2), K_num])).^2));
%             flag_occ2                  = SCCM < 5.5;                           %old vale 4 
            flag_occ                   = flag_occ1|flag_occ2;
%             [~,idx_psr]  = sort(PSR,'descend');
            %------------------------------------------------------------------------ 
             w_psr                        = PSR/sum(PSR);
             w_sim                        = sim_part/sum(sim_part);
             w_res                        = (1-params.gamma)*w_psr+params.gamma*w_sim;
            %---------------------------------------------------------------------------------------------
            mat_w        = repmat(w_res(flag_occ),[sum(flag_occ),1])+repmat(w_res(flag_occ)',[1,sum(flag_occ)]);
            mat_ww       = mat_w.*(ones(sum(flag_occ))-eye(sum(flag_occ)));
            % calculate the new position of each part
            part_pos_new                  = part_pos + part_vec;
            if sum(flag_occ)>1
               mix_dis                    = mix_distance(part_pos(flag_occ',:)); %
               mix_dis_new                = mix_distance(part_pos_new(flag_occ',:));%
               ind_pre                    = mix_dis~=0;
               ind_curr                   = mix_dis_new~=0;
               ind_all                    = ind_pre&ind_curr;
               scale_Factor               = mean(mix_dis_new(ind_all)./mix_dis(ind_all));
            else
               scale_Factor               = 1;
            end
            if scale_Factor<0.95
                scale_Factor=0.95;
            end
            if scale_Factor>1.05
                scale_Factor=1.05;
            end
            %  if frame==2
%                  resp_max_weight =resp_max;
%              else
%                  resp_max_weight=(1-learning_rate)*resp_max_weight+learning_rate*resp_max;
%              end
%             translation_vec  = round([part_disp_row; part_disp_col]*resp_max_weight'/sum(resp_max_weight)*featureRatio * currentScaleFactor*scale_Factor)';
              flag_final                   = flag_occ;% & flag_residue;            
% %              resp_max_old                 = (1 - learning_rate) * resp_max_old + learning_rate * resp_max; 
             if sum(flag_final)~=0
% %                 translation_vec  = round([part_disp_row(flag_final); part_disp_col(flag_final)]*PSR(flag_final)'/sum(PSR(flag_final))*featureRatio * currentScaleFactor*scale_Factor)';%
                  translation_vec  = round([part_disp_row(flag_final); part_disp_col(flag_final)]*w_res(flag_final)'/sum(w_res(flag_final))*featureRatio * currentScaleFactor*scale_Factor)';%
% %                  translation_vec  = round([disp_row; disp_col]*w_res'/sum(w_res)*featureRatio * currentScaleFactor*scale_Factor)';%
             else
                 translation_vec  = translation_vec_old;
             end
%              translation_vec_old  = translation_vec;
%              response_old         = response;
             
%              if frame == 110
%                  pause;
%              end
             
%             response                   = permute(response,[1 2 4 3]);
%             responsef_padded           = permute(responsef_padded,[1 2 4 3]);
%             for part_ind = 1:K_num
%                wt(part_ind)  = max(max(max(real(response(:,:,:,part_ind)))));
%             end
%             weight          = (1 - learning_rate) * weight        + learning_rate * wt;
%             mat_weight          = repmat(weight,[numel(response)/K_num,1]);
%             mat_weight          = reshape(mat_weight,size(response));
%             res_response    = sum(response.*mat_weight,4);
%             res_responsef_padded = sum(responsef_padded.*mat_weight,4);
%             [disp_row, disp_col, sind] = resp_newton(res_response, res_responsef_padded, newton_iterations, ky, kx, use_psz);
%             translation_vec = round([disp_row; disp_col]*featureRatio * currentScaleFactor*scaleFactors(sind))';
            
            % find maximum
%             if interpolate_response == 3
%                 error('Invalid parameter value for interpolate_response');
%             elseif interpolate_response == 4
%                 [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_psz);
%             else
%                 [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
%                 disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
%                 disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
%             end
            % calculate translation
%             switch interpolate_response
%                 case 0
%                     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
%                 case 1
%                     translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
%                 case 2
%                     translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
%                 case 3
%                     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
%                 case 4
%                     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
%             end
            % set the scale
%             currentScaleFactor = currentScaleFactor * scaleFactors(sind);
            currentScaleFactor = currentScaleFactor * scale_Factor;
            % adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            % update position
            old_pos = pos;
            pos = pos + translation_vec;
%             %----------------------------------------------------------------------------------
%             % calculate the global object position
%               global_pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
%             % extract features and do windowing
%               zfglb = fft2(bsxfun(@times,get_features(global_pixels,features,global_feat_params),cos_window));
%               kzfglb = gaussian_correlation(zfglb, model_xfglb, params.kernel.sigma);
%               responsef_glb = model_alphafglb.*kzfglb;%real(ifft2(model_alphafglb.*kzfglb));
%               responsefglb_padded = resizeDFT2(responsef_glb, interp_sz);
%              % response
%                response_glb = ifft2(responsefglb_padded, 'symmetric');
%              % find maximum
%                [disp_row_glb, disp_col_glb, sind] = resp_newton(response_glb, responsefglb_padded, newton_iterations, ky, kx, use_sz);
%              % calculate translation
%                translation_vec_glb  = round([disp_row_glb, disp_col_glb] * featureRatio * currentScaleFactor);
%              % estimate the final object position
%                pos         = pos + translation_vec_glb;
            % re_estimate the position of each part
            part_sz        = base_part_sz*currentScaleFactor;
            % constructed the initialized pos of each part
                if Aspect_ratio<=0.6
                  K_num        = 3;
                  idx_x        = [0 0 0];
                  idx_y        = [-1 0 1];
                  pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
                elseif Aspect_ratio>1.6
                    K_num        = 3;
                    idx_x        = [-1 0 1]';
                    idx_y        = [0 0 0]';
                    pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
                else
                      if params.pflag ==0
                          K_num        = 4;
                          idx_x        = [-0.25 0.25]'*[1 1 ];
                          idx_y        = [1 1 ]'* [-0.25 0.25];
                          pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
                      else
                          K_num            = 4;
                          idx_x            = [-0.25 0.25]'*[1 1 ];
                          idx_y            = [1 1 ]'* [-0.25 0.25];
                          pos_vec          = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz)); 
                      end
                end
%               if Aspect_ratio<=1
%                  K_num        = 2;
%                  idx_x        = [0 0 ];
%                  idx_y        = [-0.5 0.5];
%                  pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
%               else
%                  K_num        = 2;
%                  idx_x        = [-0.5 0.5]';
%                  idx_y        = [0 0 ]';
%                  pos_vec      = floor([idx_x(:) idx_y(:)].*(ones(K_num,1)*part_sz));
%               end
            part_pos =ones(K_num,1)*pos+pos_vec;
            iter = iter + 1;
        end
    end
    
    % extract the training sample feature map for the translation filter
    for part_ind=1:K_num
        parts_pixels(:,:,:,part_ind)              = get_pixels(im,part_pos(part_ind,:),round(psz*currentScaleFactor),psz);
        appearance_parts_pixels(:,:,:,part_ind)   = get_pixels(im,part_pos(part_ind,:),round(appearance_part_sz*currentScaleFactor),appearance_part_sz);
    end
    % calculate similate weight
    kappa    = 3;
    p_pixels = reshape( parts_pixels,[numel(parts_pixels)/K_num,K_num]);
    pw       = bsxfun(@minus,double(p_pixels)/255,mean(double(p_pixels)/255,2)).^2;
    ppw      = sqrt(sum(pw));
    params.ppw      = exp(-ppw/(2*kappa^2));
    % judge whether each part  is occluded
    if frame~=1
       theta           = 0.01;                                             %0.05
       dis_part        = (reshape(appearance_parts_pixels,[appearance_part_sz(1)*appearance_part_sz(2)*size(im,3),K_num])-reshape(appearance_parts_pixels_old,[appearance_part_sz(1)*appearance_part_sz(2)*size(im,3),K_num]))/255;
       dis_part_sum    = sum(dis_part.^2);
       sim_part        = exp(-theta*dis_part_sum);
       flag_occ2       = sim_part>0.2;
       appearance_parts_pixels_old(:,:,:,flag_occ') = (1 - learning_rate) *appearance_parts_pixels_old(:,:,:,flag_occ')+learning_rate*appearance_parts_pixels(:,:,:,flag_occ');
    end
    % extract features and do windowing for parts
    xl = bsxfun(@times,get_features(parts_pixels,features,global_feat_params),part_cos_win);
%     feature_dim = size(xl,3);
    % calculate the translation filter update
    new_xlf = fft2(xl);
    % calculate the denominator
%     denominator = permute(sum( new_xlf .* conj( new_xlf ) , 3 ),[1,2,4,3]); %new_denominator
     if frame==1
        params.w0           = conj(fft2(zeros(use_psz(1), use_psz(2),K_num)));
        params.eta          = 0;
        params.w_pre        = zeros(use_psz(1), use_psz(2),K_num);
    else
        params.w0           = hf_mode_alpha;
        params.w_pre        = real(ifft2(conj(hf_mode_alpha)));
        params.eta          = 5;                                 %先前的取值5;
     end
     for part_ind =1:K_num
         kf(:,:,part_ind)    = gaussian_correlation(new_xlf(:,:,:,part_ind), new_xlf(:,:,:,part_ind), params.kernel.sigma);
     end
     [alpha,bias]     = PSVKCFT_Solver_pre(params,kf);
%     for part_ind =1:K_num
%         params.ww = params.w0(:,:,part_ind);
%         [alpha(:,:,part_ind),bias(:,:,part_ind)]     =SVKCFT_Solver(params,kf(:,:,part_ind));
%     end
    %-----------------------------------------------------------------------------------------------------------
    %update model
    if frame ==1
%         denominator = new_denominator;
%         hf_bias        = bias;
        hf_mode_alpha  = alpha;
        xlf            = new_xlf;
    else
%         flag_occ                   = flag_occ1|flag_occ2;
%          hf_mode_alpha         = (1 - learning_rate) * hf_mode_alpha         + learning_rate* alpha;
%          xlf                  = (1 - learning_rate) * xlf         + learning_rate* new_xlf;%         if sum(flag_occ)==K_num
         
         hf_mode_alpha(:,:,flag_occ')         = (1 - learning_rate) * hf_mode_alpha(:,:,flag_occ')         + learning_rate* alpha(:,:,flag_occ');
         xlf(:,:,:,flag_occ')         = (1 - learning_rate) * xlf(:,:,:,flag_occ')         + learning_rate* new_xlf(:,:,:,flag_occ');
    end
%     confm                    = ifft2(hf_mode_alpha.*kf);
    if frame ==1
%        response_old                = real(confm);
       
       appearance_parts_pixels_old   = appearance_parts_pixels;
       flag_occ2                     = true(1,K_num);
    end
%      % --------------------------------------------------------------------
%      % calculate the global object position
%      global_pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
%       % extract features and do windowing
%      xlglb = bsxfun(@times,get_features(global_pixels,features,global_feat_params),cos_window);
%      % calculate the translation filter update
%      xlfglb = fft2(xlglb);
%      %Kernel Ridge Regression, calculate alphas (in Fourier domain)
%      kfglb = gaussian_correlation(xlfglb, xlfglb, params.kernel.sigma);
%      alphafglb = params.yf ./ (kfglb + lambda);   %equation for fast training
%      if frame == 1,  %first frame, train with a single image
% 			model_alphafglb = alphafglb;
% 			model_xfglb = xlfglb;
% 	 else
% 			%subsequent frames, interpolate model
% 			model_alphafglb = (1 - learning_rate) * model_alphafglb + learning_rate * alphafglb;
% 			model_xfglb = (1 - learning_rate) * model_xfglb + learning_rate * xlfglb;
%      end    
    % calculate the new target size
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position
%     positions(frame,:) = [pos target_sz];                                  %[pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    positions(frame,:)   = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]; 
    
    time = time + toc;
    
    
    %visualization
    if visualization == 1
%          rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
%          stop = update_visualization(frame, rect_position_vis);
% %          part_rect_position= [part_pos(1,[2,1]) - part_sz(1,[2,1])/2, part_sz(1,[2,1])];
% %          stop = update_visualization(frame, part_rect_position);
%         if stop, break, end  %user pressed Esc, stop early
% 

        
%         im_to_show = double(im)/255;
% %         DrawImage(im,rect_position_vis,pos,currentScaleFactor,sz,hf_mode_alpha_msf);
% %         DrawWeightMap(im,rect_position_vis,pos,currentScaleFactor,sz,1-reg_window);
%         if size(im_to_show,3) == 1
%             im_to_show = repmat(im_to_show, [1 1 3]);
%         end
%         if frame == 1
%             fig_handle = figure('Name', 'Tracking');
%             imagesc(im_to_show);
%             hold on;
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(10, 10, int2str(frame), 'color', [0 1 1]);
%             hold off;
%             axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
%         else
%             resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
%             xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
%             ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
%             sc_ind = floor((nScales - 1)/2) + 1;
%             
%             figure(fig_handle);
%             imagesc(im_to_show);
%             hold on;
%             resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind))); colormap hsv;
%             alpha(resp_handle, 0.5);
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(10, 10, int2str(frame), 'color', [0 1 1]);
%             hold off;
%         end
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
%         part_rect_position=[part_pos_new(:,[2,1]) - ones(K_num,1)*part_sz([2,1])/2, ones(K_num,1)*part_sz([2,1])];
        part_rect_color   ={'yellow','red','blue','cyan'};
        if frame == 1  %first frame, create GUI
            figure('NumberTitle','off', 'Name',['Tracker - ' ]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g');
            text_handle = text(10, 10, int2str(frame));
            set(text_handle, 'color', [0 1 1]);
            part_rect_position=[part_pos(:,[2,1]) - ones(K_num,1)*part_sz([2,1])/2, ones(K_num,1)*part_sz([2,1])];
            for j=1:K_num
                part_rect_handle{j}=rectangle('Position',part_rect_position(j,:),'EdgeColor',part_rect_color{j},'LineWidth',2,'LineStyle','--');
            end
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position)
                set(text_handle, 'string', int2str(frame));
                part_rect_position=[part_pos_new(:,[2,1]) - ones(K_num,1)*part_sz([2,1])/2, ones(K_num,1)*part_sz([2,1])];
                for j=1:K_num
                    set(part_rect_handle{j},'Position', part_rect_position(j,:));
                end
            catch
                return
            end
        end
%         
        drawnow
    end
end

fps = num_frames/time;