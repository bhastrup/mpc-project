data {
  int<lower=0> n_days;               // number of used days
  vector<lower=0>[n_days] cost;      // outcome vector
  vector[n_days] u;         // predictor vector
  vector[n_days] weights;   // model weights
}

parameters {
  real a;                    // coefficients for predictors
  real b;                    // intercept
  real<lower=0> sigma;       // error sd
}

model {
    //a ~ normal(0, 10000);
    //b ~ normal(0, 1000);
    //sigma ~ normal(0, 100);

    cost ~ normal(a * u + b, sigma);

    for (n in 1:n_days) {
        target += weights[n] * normal_lpdf(cost[n] | a*u[n] + b, sigma);
}
}
