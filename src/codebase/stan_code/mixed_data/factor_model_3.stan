data {
  int<lower=1> N;
  int<lower=0> Kb;
  int<lower=0> Kc;
  int<lower=0> number_of_groups;
  int<lower=0, upper=1> yb[N,Kb];
  vector[Kc] yc[N];
  int<lower=1, upper=number_of_groups> grp[N];
  int<lower=1> cov_num;
}

transformed data{
  int<lower=1> J = Kb+Kc; // number of variables
  int<lower=1> K = 2; // number of factors
  cov_matrix[Kb] I_Kb = diag_matrix(rep_vector(1, Kb));
  int<lower=1> mean_num = number_of_groups;
}

parameters {
  vector[J] alpha[mean_num];
  vector[1] beta_free1[cov_num]; // 2 free elements for cont effect
  vector[Kb] beta_zeros1[cov_num]; // 4 approx zero elements for cont effect

  vector[Kb] beta_free2[cov_num]; // 4 free elements for binary effect
  vector[Kc] beta_zeros2[cov_num]; // 2 approx zero elements for binary effect
  
  vector[Kb] yy[N];
  vector<lower=0>[Kc] theta[cov_num];
  cov_matrix[Kb] Omega_cov[cov_num];
  cholesky_factor_corr[K] L_Phi[cov_num];
}

transformed parameters{
  // vector[J] yy[N];
  matrix[J,K] beta_pool[cov_num];
  cov_matrix[Kc] Marg_cov_cont_pool[cov_num];
  cov_matrix[Kb] Marg_cov_bin_pool[cov_num];
  corr_matrix[K] Phi_cov[cov_num];
  matrix[J,J] betabeta[cov_num];

  for (g in 1:cov_num){
  for(j in 1:J) {
    for (k in 1:K) beta_pool[g, j,k] = 0;
  }

  beta_pool[g, 1, 1] = 1; // fix first loading to 1
  // set the free elements
  beta_pool[g, 2, 1] = beta_free1[g,1];
  beta_pool[g, (1+Kc):J, 2] = beta_free2[g,] ;
  
  // set the zero elements
  beta_pool[g, (Kc+1):J, 1] = beta_zeros1[g, ];
  beta_pool[g, 1:Kc, 2] = beta_zeros2[g, ];

  Phi_cov[g, ] = multiply_lower_tri_self_transpose(L_Phi[g, ]);
  betabeta[g, ] =  beta_pool[g, ] * to_matrix(Phi_cov[g, ]) * beta_pool[g,]'; 
  
  Marg_cov_cont_pool[g, ] = betabeta[g, 1:Kc,1:Kc] + diag_matrix(theta[g, ]);
  Marg_cov_bin_pool[g, ] = betabeta[g, (1+Kc):,(1+Kc):] + Omega_cov[g, ];
  }
}

model {
  for (g in 1:cov_num){
  beta_free1[g,] ~ normal(0, 1);
  beta_free2[g,] ~ normal(0, 1);
  beta_zeros1[g,] ~ normal(0, 0.1);
  beta_zeros2[g,] ~ normal(0, 0.1);
  theta[g,] ~ cauchy(0,1);
  L_Phi[g,] ~ lkj_corr_cholesky(2);
  Omega_cov[g,] ~ inv_wishart(Kb+6, I_Kb);
  }
  for (g in 1:mean_num) to_vector(alpha[g,]) ~ normal(0, 10);
  
  if (cov_num==1){
    for (n in 1:N) yc[n,] ~ multi_normal(alpha[grp[n],:Kc], Marg_cov_cont_pool[1, ]);
    for (n in 1:N) yy[n,] ~ multi_normal(alpha[grp[n],(Kc+1):], Marg_cov_bin_pool[1, ]);
  }
  else if (cov_num>1){
    for (g in 1:cov_num){
      for (n in 1:N) yc[n,] ~ multi_normal(alpha[grp[n],:Kc], Marg_cov_cont_pool[g, ]);
      for (n in 1:N) yy[n,] ~ multi_normal(alpha[grp[n],(Kc+1):], Marg_cov_bin_pool[g, ]);
    }
  }
  for (j in 1:Kb) yb[, j] ~ bernoulli_logit(yy[,j]);
}

generated quantities{
  matrix[J,K] beta[number_of_groups];
  matrix[Kc,K] beta_c[number_of_groups];
  matrix[Kb,K] beta_b[number_of_groups];
  cov_matrix[Kc] Marg_cov_cont[number_of_groups];
  cov_matrix[Kb] Marg_cov_bin[number_of_groups];
  vector[Kc] alpha_c[number_of_groups];
  vector[Kb] alpha_b[number_of_groups];
  for (g in 1:number_of_groups){
    if (cov_num==1){
      beta[g,] = beta_pool[1,];
      Marg_cov_cont[g,] = Marg_cov_cont_pool[1,];
      Marg_cov_bin[g,] = Marg_cov_bin_pool[1,];
    }
    else if (cov_num>1){
      beta[g,] = beta_pool[g,];
      Marg_cov_cont[g,] = Marg_cov_cont_pool[g,];
      Marg_cov_bin[g,] = Marg_cov_bin_pool[g,];
    }
    beta_c[g,] = beta[g,:Kc,:];
    beta_b[g,] = beta[g,(Kc+1):,:];
    alpha_c[g,] = alpha[g,:Kc];
    alpha_b[g,] = alpha[g,(Kc+1):];
    }
}
