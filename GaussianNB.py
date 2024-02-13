import numpy as np

class GaussianNB_pp:

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
		
    def logprior(self, class_ind):
        return np.log(self.class_priors_[class_ind])
	    
    def loglikelihood(self, Xi, class_ind):
        # mu: mean, var: variance, Xi: sample (a row of X)    
        # Get the class mean 
        mu = self.theta_[class_ind,:]
        # Get the class variance 
        var =self.var_[class_ind,:]
        # Write the Gaussian Likelihood expression
        GaussLikelihood = (1 /( var*(np.sqrt(2 * np.pi)))) * np.exp(-0.5 * ((Xi - mu) ** 2 / var))
        # Take the log of GaussLikelihood
        logGaussLikelihood = np.log(GaussLikelihood)
        # Return loglikelihood of the sample. Now you will use the "naive" 
        # part of the naive bayes. 

        return logGaussLikelihood
			
    def posterior(self, Xi, class_ind):
        logprior = self.logprior(class_ind)				
        loglikelihood = self.loglikelihood(Xi, class_ind)
        # Return posterior
        return logprior+np.sum(loglikelihood)

    def fit(self, X, y):
		# Number of samples, number of features
        n_samples, n_features = X.shape
		# Get the unique classes
        self.classes_ = np.unique(y)
		# Number of classes
        n_classes = len(self.classes_)

		# Initialize attributes for each class
		# Feature means for each class, shape (n_classes, n_features)
        self.theta_ = np.zeros((n_classes,n_features))
		# Feature variances for each class shape (n_classes, n_features)
        self.var_ = np.zeros((n_classes,n_features))
		# Class priors shape (n_classes,)
        self.class_priors_=np.zeros(n_classes)
        
        # Calculate class means, variances and priors
        for c_ind, c_id in enumerate(self.classes_):
            # Get the samples that belong to class c_id
            X_class = X[y == c_id, :]
            # Mean of the each feature that belongs to class c_id
            self.theta_[c_ind, :] = np.mean(X_class, axis=0)
            # Calculate the variance of each feature that belongs to c_id		
            self.var_[c_ind, :] = np.var(X_class,axis=0) +self.var_smoothing
            # Calculate the priors for each class
            self.class_priors_[c_ind] = np.sum(y==c_id)/len(y)

                  
    def predict(self, X):
        y_pred = []
        for Xi in X: # Calculate posteriors for each sample
            posteriors = []	# For saving posterior values for each class
			# Calculate posterior probability for each class
            for class_ind in self.classes_:
				# Calculate posterior
                sample_posterior = self.posterior(Xi, class_ind)
				# Append the posterior value of this class to posteriors
                posteriors.append(sample_posterior)
                # print(sample_posterior[0])
			# Get the class that has the highest posterior prob. and
			# append the prediction for this sample to y_pred
            y_pred.append(self.classes_[np.argmax(posteriors)])					
            			
        return y_pred # Return predictions for all samples

	
	
