data{
  int<lower=0> N;
  int<lower=0> K;
  int<lower=0> Kb;
  int<lower=0> Kc;
  int<lower=0, upper=1> yb[N,Kb];
  vector[Kc] yc[N];
  int<lower=0> number_of_groups;
  int<lower=1, upper=number_of_groups> grp[N];
  int<lower=1> cov_num;
}

transformed data {
  matrix[Kc, Kc] I = diag_matrix(rep_vector(1, Kc));
  int<lower=1> mean_num = number_of_groups;
}

parameters {
  vector[Kb] zb[N];
  cholesky_factor_corr[K] L_R[cov_num];  // first continuous, then binary
  vector<lower=0>[Kc] sigma[cov_num];
  vector[K] alpha[mean_num];
}

transformed parameters{
  matrix[N, Kb] z;
  vector[Kc] alpha_c[mean_num];
  vector[Kb] alpha_b[mean_num];
  matrix[Kc, Kc] L_inv[cov_num];
  vector[Kc] resid;
  for (g in 1: mean_num){
  alpha_c[g,] = head(alpha[g,], Kc);
  alpha_b[g,] = tail(alpha[g,], Kb);
  } 
    if (cov_num==1){
      for (g in 1:cov_num){
      L_inv[g,] = mdivide_left_tri_low(diag_pre_multiply(sigma[g,], L_R[g, 1:Kc, 1:Kc]), I);
      }
      for (n in 1:N){
        resid = L_inv[1,] * (yc[n] - alpha_c[grp[n],]);
        z[n,] = transpose(alpha_b[grp[n],] + tail(L_R[1,] * append_row(resid, zb[n]), Kb));
      }
    }
    else if (cov_num>1){
      for (g in 1:cov_num){
        L_inv[g,] = mdivide_left_tri_low(diag_pre_multiply(sigma[g,], L_R[g, 1:Kc, 1:Kc]), I);
      }
      for (n in 1:N){
        resid = L_inv[grp[n],] * (yc[n] - alpha_c[grp[n],]);
        z[n,] = transpose(alpha_b[grp[n],] + tail(L_R[grp[n],] * append_row(resid, zb[n]), Kb));
      }
    }
}

model{
  for (g in 1:cov_num){
   L_R[g,] ~ lkj_corr_cholesky(2);
   sigma[g,] ~ cauchy(0,2.5);
  }
  for (g in 1: mean_num) alpha[g,] ~ normal(0,10); 
  if (cov_num==1){
    for (n in 1:N){
      yc[n,] ~ multi_normal_cholesky(alpha_c[grp[n],], diag_pre_multiply(sigma[1, ], L_R[1, 1:Kc, 1:Kc]));
    }
  }
  else if (cov_num>1){
    for (n in 1:N){
      yc[n,] ~ multi_normal_cholesky(alpha_c[grp[n],], diag_pre_multiply(sigma[grp[n], ], L_R[grp[n], 1:Kc, 1:Kc]));
    }
  }
  for (n in 1:N) zb[n] ~ normal(0,1);
  for (k in 1:Kb) yb[,k] ~ bernoulli_logit(z[,k]);
}


generated quantities{
  vector[K] yy[N];
  matrix[K,K] R[number_of_groups];
  vector[K] full_sigma[number_of_groups];
  matrix[K,K] Marg_cov[number_of_groups];
  cov_matrix[Kc] Marg_cov_cont[number_of_groups];
  cov_matrix[Kb] Marg_cov_bin[number_of_groups];
  for (g in 1:number_of_groups){
    if (cov_num==1){
      full_sigma[g,] = append_row(sigma[1,], rep_vector(1, Kb));
      R[g,] = multiply_lower_tri_self_transpose(L_R[1,]);
      Marg_cov[g, ] = multiply_lower_tri_self_transpose(diag_pre_multiply(full_sigma[1,],L_R[1,])); 
    }
    else if (cov_num>1){
      full_sigma[g,] = append_row(sigma[g,], rep_vector(1, Kb));
      R[g,] = multiply_lower_tri_self_transpose(L_R[g,]);
      Marg_cov[g, ] = multiply_lower_tri_self_transpose(diag_pre_multiply(full_sigma[g,],L_R[g,])); 
    }
    Marg_cov_cont[g,] = Marg_cov[g, :Kc, :Kc];
    Marg_cov_bin[g,] = Marg_cov[g, (Kc+1):, (Kc+1):];
  }
  for (k in 1:Kc) yy[,k] = yc[,k]; 
  for (n in 1:N){
    for (k in 1:Kb) yy[n,(k+Kc)] = z[n,k]; 
  }
  
}
