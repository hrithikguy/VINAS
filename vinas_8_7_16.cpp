#include <iostream>
#include <map>
#include <string.h>
#include <iterator>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <dirent.h>
#include <sstream>
#include <vector>
#include <omp.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdlib>
#include <math.h>
#include <string>
#include <errno.h>
#include <algorithm>
#include <float.h>
#include <utility>



using namespace std;

#define MAXIT 100
#define EPS 3.0e-7
#define FPMIN 1.0e-30
#define SIGMA_TINY 1.0e-20

#ifndef M_LN_SQRT_2PI
#define M_LN_SQRT_2PI	0.918938533204672741780329736406	/* log(sqrt(2*pi)) */
#endif

#ifndef M_LN_2PI
#define M_LN_2PI	1.837877066409345483560659472811	/* log(2*pi) */
#endif

double R_D_exp (double x, bool log_p) {
	return (log_p	? (x)	: exp(x));
}


bool reverseSortFirst (vector<double> a, vector<double> b) {
	return a[0] > b[0];
}

bool sortVectorsSecond (vector<double> a, vector<double> b) {
	return a[1] < b[1];
}

double gammln(double xx) {
	double x,y,tmp,ser;
	static double cof[6]={76.18009172947146,-86.50532032941677,
												24.01409824083091,-1.231739572450155,
												0.1208650973866179e-2,-0.5395239384953e-5};
	int j;
	
	y=x=xx;
	tmp=x+5.5;
	tmp -= (x+0.5)*log(tmp);
	ser=1.000000000190015;
	for (j=0;j<=5;j++) ser += cof[j]/++y;
	return -tmp+log(2.5066282746310005*ser/x);
}

double stirlerr(double n) {

#define S0 0.083333333333333333333       /* 1/12 */
#define S1 0.00277777777777777777778     /* 1/360 */
#define S2 0.00079365079365079365079365  /* 1/1260 */
#define S3 0.000595238095238095238095238 /* 1/1680 */
#define S4 0.0008417508417508417508417508/* 1/1188 */

/*
  error for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
*/
    const static double sferr_halves[31] = {
	0.0, /* n=0 - wrong, place holder only */
	0.1534264097200273452913848,  /* 0.5 */
	0.0810614667953272582196702,  /* 1.0 */
	0.0548141210519176538961390,  /* 1.5 */
	0.0413406959554092940938221,  /* 2.0 */
	0.03316287351993628748511048, /* 2.5 */
	0.02767792568499833914878929, /* 3.0 */
	0.02374616365629749597132920, /* 3.5 */
	0.02079067210376509311152277, /* 4.0 */
	0.01848845053267318523077934, /* 4.5 */
	0.01664469118982119216319487, /* 5.0 */
	0.01513497322191737887351255, /* 5.5 */
	0.01387612882307074799874573, /* 6.0 */
	0.01281046524292022692424986, /* 6.5 */
	0.01189670994589177009505572, /* 7.0 */
	0.01110455975820691732662991, /* 7.5 */
	0.010411265261972096497478567, /* 8.0 */
	0.009799416126158803298389475, /* 8.5 */
	0.009255462182712732917728637, /* 9.0 */
	0.008768700134139385462952823, /* 9.5 */
	0.008330563433362871256469318, /* 10.0 */
	0.007934114564314020547248100, /* 10.5 */
	0.007573675487951840794972024, /* 11.0 */
	0.007244554301320383179543912, /* 11.5 */
	0.006942840107209529865664152, /* 12.0 */
	0.006665247032707682442354394, /* 12.5 */
	0.006408994188004207068439631, /* 13.0 */
	0.006171712263039457647532867, /* 13.5 */
	0.005951370112758847735624416, /* 14.0 */
	0.005746216513010115682023589, /* 14.5 */
	0.005554733551962801371038690  /* 15.0 */
    };
    double nn;

    if (n <= 15.0) {
	nn = n + n;
	if (nn == (int)nn) return(sferr_halves[(int)nn]);
	return(gammln(n + 1.) - (n + 0.5)*log(n) + n - M_LN_SQRT_2PI);
    }

    nn = n*n;
    if (n>500) return((S0-S1/nn)/n);
    if (n> 80) return((S0-(S1-S2/nn)/nn)/n);
    if (n> 35) return((S0-(S1-(S2-S3/nn)/nn)/nn)/n);
    /* 15 < n <= 35 : */
    return((S0-(S1-(S2-(S3-S4/nn)/nn)/nn)/nn)/n);
}

double bd0(double x, double np) {
    double ej, s, s1, v;
    int j;

    if (isnan(x) || isnan(np) || np == 0.0) {
    	perror("Error: bd0 given an argument that is not a number.\n");
			exit(1);
		}

    if (fabs(x-np) < 0.1*(x+np)) {
	v = (x-np)/(x+np);  // might underflow to 0
	s = (x-np)*v;/* s using v -- change by MM */
	if(fabs(s) < DBL_MIN) return s;
	ej = 2*x*v;
	v = v*v;
	for (j = 1; j < 1000; j++) { /* Taylor series; 1000: no infinite loop
					as |v| < .1,  v^2000 is "zero" */
	    ej *= v;// = v^(2j+1)
	    s1 = s+ej/((j<<1)+1);
	    if (s1 == s) /* last term was effectively 0 */
		return s1 ;
	    s = s1;
	}
    }
    /* else:  | x - np |  is not too small */
    return(x*log(x/np)+np-x);
}

double dbinom (double x, double n, double p, bool log_p) {
	
	// Error checking
	if ((p < 0) || (p > 1)) {
		perror("p must be between 0 and 1\n");
		exit(1);
	}
	if (x < 0) {
		perror("x must be >=0\n");
		exit(1);
	}
	if (n < x) {
		perror("x must be <= than the binomial denominator\n");
		exit(1);
	}
	double q = 1 - p;
	double lf, lc;
	
	if (p == 0) return((x == 0) ? (log_p ? 0. : 1.) : (log_p ? -DBL_MAX : 0.));
  if (q == 0) return((x == n) ? (log_p ? 0. : 1.) : (log_p ? -DBL_MAX : 0.));
  
  if (x == 0) {
		if(n == 0) return (log_p ? 0. : 1.);
		lc = (p < 0.1) ? -bd0(n,n*q) - n*p : n*log(q);
		return( (log_p	?  (lc)	 : exp(lc)) );
  }
  if (x == n) {
		lc = (q < 0.1) ? -bd0(n,n*p) - n*q : n*log(p);
		return( (log_p	?  (lc)	 : exp(lc)) );
  }
  if (x < 0 || x > n) return( (log_p ? -DBL_MAX : 0.) );
  
  /* n*p or n*q can underflow to zero if n and p or q are small.  This
		 used to occur in dbeta, and gives NaN as from R 2.3.0.  */
	lc = stirlerr(n) - stirlerr(x) - stirlerr(n-x) - bd0(x,n*p) - bd0(n-x,n*q);

	/* f = (M_2PI*x*(n-x))/n; could overflow or underflow */
	/* Upto R 2.7.1:
	 * lf = log(M_2PI) + log(x) + log(n-x) - log(n);
	 * -- following is much better for  x << n : */
	lf = M_LN_2PI + log(x) + log1p(- x/n);

	return R_D_exp((lc - 0.5*lf), log_p);
}

double betacf(double a, double b, double x) {
	int m,m2;
	double aa,c,d,del,h,qab,qam,qap;
	
	qab=a+b;
	qap=a+1.0;
	qam=a-1.0;
	c=1.0;
	d=1.0-qab*x/qap;
	if (fabs(d) < FPMIN) {
		d=FPMIN;
	}
	d=1.0/d;
	h=d;
	for (m=1;m<=MAXIT;m++) {
		m2=2*m;
		aa=m*(b-m)*x/((qam+m2)*(a+m2));
		d=1.0+aa*d;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=1.0+aa/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0/d;
		h *= d*c;
		aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
		d=1.0+aa*d;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=1.0+aa/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0/d;
		del=d*c;
		h *= del;
		if (fabs(del-1.0) < EPS) break;
	}
	if (m > MAXIT) {
		perror("a or b too big, or MAXIT too small in betacf");
		exit(1);
	}
	return h;
}

double betai(double a, double b, double x) {

	double bt;

	if (x < 0.0 || x > 1.0) {
		perror("Bad x in routine betai");
		exit(1);
	}
	
	if (x == 0.0 || x == 1.0) {
		bt=0.0;
	} else {
		bt = exp(gammln(a+b)-gammln(a)-gammln(b)+a*log(x)+b*log(1.0-x));
	}
	
	if (x < (a+1.0)/(a+b+2.0)) {
		return bt*betacf(a,b,x)/a;
	} else {
		return 1.0-bt*betacf(b,a,1.0-x)/b;
	}
}

double do_search(double y, double *z, double p, double n, double pr, double incr) {
	if (*z >= p) {
		/* search to the left */
		for(;;) {
			double newz;
	    // if (y == 0 || (newz = pbinom(y - incr, n, pr)) < p) {
	    if (y == 0 || (newz = betai(y - incr, n-(y-incr)+1, pr)) < p) {
	    	return y;
	    }
	    y = fmax(0, y - incr);
	    *z = newz;
		}
	} else {
		/* search to the right */
		for(;;) {
			y = fmin(y + incr, n);
			if (y == n || (*z = betai(y, n-y+1, pr)) >= p) {
				return y;
			}
		}
	}
}

double round_to_digits(double value, int digits)
{
	if (value == 0.0) {
		return 0.0;
	} else if (value > 0.0 && value < 0.05) {
		return 0.0;
	} else {
    double factor = pow(10.0, digits - ceil(log10(fabs(value))));
    return round(value * factor) / factor;   
  }
}

double final_rounding(double value) {
	if (isnan(value)) {
		return 0.0;
	} else {
		double factor = pow(10.0, 3);
		double res = round(value * factor) / factor;
  	return res;
  }
}

double R_D_Lval(double p, bool lower_tail) {
	return (lower_tail ? (p) : (0.5 - (p) + 0.5));
}

double R_DT_qIv(double p, bool lower_tail, bool log_p) {
	return (log_p ? (lower_tail ? exp(p) : - expm1(p)) : R_D_Lval(p, lower_tail));
}

double R_D_Cval(double p, bool lower_tail) {
	return (lower_tail ? (0.5 - (p) + 0.5) : (p));
}

double R_DT_CIv(double p, bool lower_tail, bool log_p) {
	return (log_p ? (lower_tail ? -expm1(p) : exp(p)) : R_D_Cval(p, lower_tail));
}

double qnorm(double p, double mu, double sigma, bool lower_tail, bool log_p) {
	double p_, q, r, val;
	
	// Error checking
	if (isnan(p) || isnan(mu) || isnan(sigma)) {
		perror("Error: qnorm given an argument that is not a number.\n");
		exit(1);
	}
	
	if (sigma < SIGMA_TINY) {
		sigma = SIGMA_TINY;
	}
	
	if (sigma < 0) {
		perror("Error: qnorm given a negative sigma\n");
		exit(1);
	} else if (sigma == 0) {
		return mu;
	}
	
	p_ = R_DT_qIv(p, true, false);/* real lower_tail prob. p */
  q = p_ - 0.5;
  
  if (fabs(q) <= .425) {/* 0.075 <= p <= 0.925 */
  	r = .180625 - q * q;
  	val =
            q * (((((((r * 2509.0809287301226727 +
                       33430.575583588128105) * r + 67265.770927008700853) * r +
                     45921.953931549871457) * r + 13731.693765509461125) * r +
                   1971.5909503065514427) * r + 133.14166789178437745) * r +
                 3.387132872796366608)
            / (((((((r * 5226.495278852854561 +
                     28729.085735721942674) * r + 39307.89580009271061) * r +
                   21213.794301586595867) * r + 5394.1960214247511077) * r +
                 687.1870074920579083) * r + 42.313330701600911252) * r + 1.);
  } else { /* closer than 0.075 from {0,1} boundary */
  	/* r = min(p, 1-p) < 0.075 */
		if (q > 0) {
	    r = R_DT_CIv(p, lower_tail, log_p);/* 1-p */
		} else {
	    r = p_;/* = R_DT_Iv(p) ^=  p */
	  }
	  
	  r = sqrt(- ((log_p &&
		     ((lower_tail && q <= 0) || (!lower_tail && q > 0))) ?
		    p : /* else */ log(r)));
        /* r = sqrt(-log(r))  <==>  min(p, 1-p) = exp( - r^2 ) */
        
  	if (r <= 5.) { /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
				r += -1.6;
				val = (((((((r * 7.7454501427834140764e-4 +
									 .0227238449892691845833) * r + .24178072517745061177) *
								 r + 1.27045825245236838258) * r +
								3.64784832476320460504) * r + 5.7694972214606914055) *
							r + 4.6303378461565452959) * r +
						 1.42343711074968357734)
						/ (((((((r *
										 1.05075007164441684324e-9 + 5.475938084995344946e-4) *
										r + .0151986665636164571966) * r +
									 .14810397642748007459) * r + .68976733498510000455) *
								 r + 1.6763848301838038494) * r +
								2.05319162663775882187) * r + 1.);
		} else { /* very close to  0 or 1 */
				r += -5.;
				val = (((((((r * 2.01033439929228813265e-7 +
									 2.71155556874348757815e-5) * r +
									.0012426609473880784386) * r + .026532189526576123093) *
								r + .29656057182850489123) * r +
							 1.7848265399172913358) * r + 5.4637849111641143699) *
						 r + 6.6579046435011037772)
						/ (((((((r *
										 2.04426310338993978564e-15 + 1.4215117583164458887e-7)*
										r + 1.8463183175100546818e-5) * r +
									 7.868691311456132591e-4) * r + .0148753612908506148525)
								 * r + .13692988092273580531) * r +
								.59983220655588793769) * r + 1.);
		}
		if(q < 0.0) {
	    val = -val;
	  }
    /* return (q >= 0.)? r : -r ;*/
  }
  return mu + sigma * val;
}

double pbinom (int x, int n, double p) {
	double result = 0;
	for (int i = x+1; i <= n; i++) {
		result += dbinom((double)i, (double)n, p, false);
	}
	return result;
}

// Benjamini-Hochberg implementation
// Assumes we always use n = length(p)
vector<double> bh_adjust (vector<double> p) {
	// Save only the non-NaNs
	// vector<double> p_new;
	vector<vector<double> > p_ind;
	int p_ind_index = 0;
	for (unsigned int i = 0; i < p.size(); i++) {
		if (!isnan(p[i])) {
			vector<double> temp;
			temp.push_back(p[i]);
			temp.push_back((double)p_ind_index);
			p_ind_index++;
			p_ind.push_back(temp);
		}
	}
	
	// Sort p_ind by decreasing order
	sort(p_ind.begin(), p_ind.end(), reverseSortFirst);
	
	// Do the n/i calculation
	// Dot product between seq and p_ind
	for (unsigned int i = 0; i < p_ind.size(); i++) {
		
		unsigned int i_inv = p_ind.size() - i;
		double temp = (double)p.size()/(double)i_inv;
		p_ind[i][0] = temp*(p_ind[i][0]);
	}
	
	// Cumulative minimum
	double curmin;
	for (unsigned int i = 0; i < p_ind.size(); i++) {
	
		// Initialization
		if (i == 0) {
			curmin = p_ind[i][0];
		}
	
		double this_iter = fmin(curmin, p_ind[i][0]);
		
		// Update curmin if necessary
		if (p_ind[i][0] < curmin) {
			curmin = p_ind[i][0];
		}
		
		// Don't let this value exceed 1
		if (this_iter > 1) {
			this_iter = 1.0;
		}
		
		p_ind[i][0] = this_iter;
	}
	
	sort(p_ind.begin(), p_ind.end(), sortVectorsSecond);
	
	// Save the firsts
	vector<double> retval;
	for (unsigned int i = 0; i < p_ind.size(); i++) {
		retval.push_back(p_ind[i][0]);
	}
	return retval;
}

// qbinom implementation
double qbinom(double p, double n, double pr, bool lower_tail, bool log_p) {
	double q, mu, sigma, gamma, z, y;
	
	// Error checking
	if (isnan(p) || isnan(n) || isnan(pr)) {
		perror("Error: Input to qbinom is NaN\n");
		exit(1);
	}
	if (pr < 0 || pr > 1 || n < 0) {
		perror("Error: Input to qbinom is out of bounds\n");
		exit(1);
	}
	
	if (pr == 0. || n == 0) return 0.;
	
	q = 1 - pr;
	if(q == 0.) return n; /* covers the full range of the distribution */
	mu = n * pr;
	sigma = sqrt(n * pr * q);
	gamma = (q - pr) / sigma;
	
	if (!lower_tail || log_p) {
		p = R_DT_qIv(p, true, false);
		if (p == 0.) return 0.;
		if (p == 1.) return n;
  }
  if (p + 1.01*DBL_EPSILON >= 1.) {
  	return n;
  }
  z = qnorm(p, 0., 1., /*lower_tail*/true, /*log_p*/false);
  y = floor(mu + sigma * (z + gamma * (z*z - 1) / 6) + 0.5);
  
  if (y > n) { /* way off */
  	y = n;
  }
  
  z = betai(y, n-y+1, pr);
  
  /* fuzz to ensure left continuity: */
  p *= 1 - 64*DBL_EPSILON;
  
  if (n < 1e5) {
  	return do_search(y, &z, p, n, pr, 1);
  } else { /* Otherwise be a bit cleverer in the search */
  	double incr = floor(n * 0.001), oldincr;
  	do {
	    oldincr = incr;
	    y = do_search(y, &z, p, n, pr, incr);
	    incr = fmax(1, floor(incr/100));
		} while(oldincr > 1 && incr > n*1e-15);
		return y;
	}
}



float process_covariate(float n) {
        if (n >= 0.0000001) {
                return n;
        }
        else {
                return 0;
        }
}


double min(double a, double b) {
	if (a < b) {
		return a;
	} else {
		return b;
	}
}

vector<string> split(string str, char delimiter) {
  vector<string> internal;
  stringstream ss(str);
  string tok;
  
  while(getline(ss, tok, delimiter)) {
    internal.push_back(tok);
  }
  
  return internal;
}


int non_default_vinas(char *covariants_directory, char *input_mutations, char *input_annotations)
{
	map<string, int> hashmap;
	map<string, string> predictions;
	

	ifstream mutations;
	mutations.open (input_mutations);
	string line_init;
	char newline_init[1000];
	char *element1_init;
	char *element2_init;
	char *element3_init;
	char *element4_init;
	char key_init[1000];
	vector<string> elements_init;
	ifstream names_file;
	strcpy(key_init, "ls ");
	strcat(key_init, covariants_directory);
	strcat(key_init, " > files");
	system(key_init);
	names_file.open("files");
	int filecounter = 0;
	char covariant_files[1000][1000];
	while(getline(names_file, line_init)) {
		elements_init = split(line_init, '\t');
		strcpy(newline_init, line_init.c_str());
		strcpy(covariant_files[filecounter], newline_init);
		filecounter++;
	}
	cout << filecounter << endl;



	int counter = 0;
	while (getline(mutations, line_init)) {
		elements_init.clear();
		elements_init = split(line_init, '\t');
		strcpy(key_init, elements_init[0].c_str());
		strcat(key_init, "-");
		strcat(key_init, elements_init[1].c_str());
		counter++;
		hashmap[key_init] = 1;
	}
	cout << counter << endl;
	mutations.close();
	string line;
	string line2;
	vector<string> elements;
	vector<string> elements2;
        ifstream covariant;
        ofstream liblinear;
	ifstream liblinear_out;
	char filename[100];
	char filename2[100];
	int i;
	char key[1000];
	char key2[1000];
	#pragma omp parallel for private(counter, key2, elements2, filename, line2, liblinear_out, key, filename2, line, elements, covariant, liblinear)
	for (i = 0; i < filecounter; ++i) {
		strcpy(filename, covariants_directory);
		strcat(filename, "/");
		strcat(filename, covariant_files[i]);
		fflush(stdout);
		covariant.open(filename);
		strcpy(filename2, "");
		strcat(filename2, "./liblinear_format/");
		strcat(filename2, covariant_files[i]);
		liblinear.open(filename2);
		while(getline(covariant, line)){
			elements.clear();
       			elements = split(line, '\t');
			strcpy(key, elements[0].c_str());
			strcat(key, "-");
			strcat(key, elements[1].c_str());
			if(hashmap.find(key) != hashmap.end()) {
				liblinear << "1";
			}
			else {
				liblinear << "0";
			}
			for (int i = 3; i < elements.size(); ++i) {
				
				liblinear << " " <<  i-2 << ":" << process_covariate(atof(elements[i].c_str()));
       		        }
			liblinear << endl;
			elements.clear();
		}	
		covariant.close();
		liblinear.close();
		strcpy(key, "train -q -s 0 liblinear_format/");
		strcat(key, covariant_files[i]);
		strcat(key, " liblinear_format/");
		strcat(key, covariant_files[i]);
		strcat(key, ".model");
		fflush(stdout);
		system(key);
		strcpy(key, "predict -q -b 1 liblinear_format/");
		strcat(key, covariant_files[i]);
		strcat(key, " liblinear_format/");
		strcat(key, covariant_files[i]);
		strcat(key, ".model liblinear_format/");
		strcat(key, covariant_files[i]);
		strcat(key, ".out");
                fflush(stdout);
		system(key);
		strcpy(key, "./covariants_head/");
		strcat(key, covariant_files[i]);
		covariant.open(key);
		strcpy(key, "./liblinear_format/");
		strcat(key, covariant_files[i]);
		strcat(key, ".out");
		liblinear_out.open(key);
		counter = 0;
		getline(liblinear_out, line2);
		#pragma omp critical
		while (getline(liblinear_out, line2) && getline(covariant, line)) {
			elements.clear();
			elements = split(line, '\t');
			strcpy(key, elements[0].c_str());
			strcat(key, "-");
			strcat(key, elements[1].c_str());
			elements2.clear();
			elements2 = split(line2, ' ');
			strcpy(key2, elements2[2].c_str());
			predictions[key] = key2;
		}
		covariant.close();
		liblinear_out.close();
	}
	counter = 0;
	for (std::map<string, string>::iterator it=predictions.begin(); it!=predictions.end(); ++it) {
    		counter++;
	}
	cout << counter << endl;
	ifstream annotations;
	ofstream pred_values;
	pred_values.open("predictions_only");
	annotations.open("input_annotations.bed");
	while(getline(annotations, line)) {
		elements.clear();
		elements = split(line, '\t');
		strcpy(key, elements[0].c_str());
		strcat(key, "-");
		strcat(key, elements[1].c_str());
		printf("%s\n", key);
		if(predictions.find(key) != predictions.end()) {
			pred_values << predictions[key] << endl;
		}
	}
}



int main(int argc, char **argv) {
	int dflag = 0;
  	char *cvalue = NULL;
	char *avalue = NULL;
	char *vvalue = NULL;
	char *ovalue = NULL;
  	char *nvalue = NULL;
  	int index;
  	int c;

  	opterr = 0;

  	while ((c = getopt (argc, argv, "n:dc:a:v:o:")) != -1)
    		switch (c)
      		{
      		case 'a':
        		avalue = optarg;
        		break;
      		case 'v':
        		vvalue = optarg;
        		break;
                case 'o':
                        ovalue = optarg;
                        break;
                case 'n':
                        nvalue = optarg;
                        break;
                case 'd':
                        dflag = 1;
                        break;
      		case 'c':
        		cvalue = optarg;
        		break;
      		case '?':
        		if (optopt == 'c' || optopt == 'n' || optopt == 'a' || optopt == 'v' || optopt == 'o')
          			fprintf (stderr, "Option -%c requires an argument.\n", optopt);
	        	else if (isprint (optopt))
	          		fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	       		else
	 	       		fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        		return 1;
      		default:
        		abort ();
      		}


  	for (index = optind; index < argc; index++)
    		printf ("Non-option argument %s\n", argv[index]);
	double normalize;

	if (avalue == NULL || vvalue == NULL || ovalue == NULL) {
		fprintf(stderr, "annotations, variants, and output files must be provided\n");
		return 1;
	}

	if (dflag == 1 && cvalue != NULL) {
		fprintf(stderr, "do not provide covariants directory if you choose the default option\n");
		return 1;
	}

	if (dflag == 0 && cvalue == NULL) {
		fprintf(stderr, "covariants directory must be provided if you don't choose the default option\n");
		return 1;
	}
	
	if (nvalue == NULL) {
		normalize = 2.0;
	} else {
		normalize = atof(nvalue);
		if (normalize == 0.0 || normalize < 0) {
			fprintf(stderr, "Normalization value must be a numeric value greater than 0.\n");
			return 0;
		}	
	}	
	ifstream annotation_file(avalue);
	if (!annotation_file) {
		fprintf(stderr, "could not open annotations file\n");
	}
	ifstream variant_file(vvalue);
	if (!variant_file) {
		fprintf(stderr, "could not open variants file\n");
	}
	if (cvalue != NULL) {
		struct stat info;
		if( stat( cvalue, &info ) != 0 ) {
    			printf( "cannot access %s\n", cvalue );
			return 1;
		}
		else if( info.st_mode & S_IFDIR ) {
    			printf( "%s is a directory\n", cvalue );
		}
		else {
    			printf( "%s is not a directory\n", cvalue );
			return 1;
		}
	}	
       	printf("reading in mutations file\n"); 
        string line_init;
	string line_init2;
        char key_init[1000];
        vector<string> elements_init;
	map<string, int> variants_map;       
        int counter = 0;
        while (getline(variant_file, line_init)) {
                elements_init.clear();
                elements_init = split(line_init, '\t');
                strcpy(key_init, elements_init[0].c_str());
                strcat(key_init, "-");
                strcat(key_init, elements_init[1].c_str());
                counter++;
                if(variants_map.find(key_init) != variants_map.end()) {
                	variants_map[key_init] += 1;
		} else {
			variants_map[key_init] = 1;
		}
        }
	variant_file.close();
	ofstream pmodel("intersection_counts");
	ofstream pfmodel("intersections.bed");
	
	int i;
	int start;
	int stop;
	char temp[100];
	char temp2[100];
	int mutation_count;
	vector <int> mutation_counts;
	printf("getting intersections\n");
	while (getline(annotation_file, line_init)) {
		elements_init.clear();
		elements_init = split(line_init, '\t');
		start = atoi(elements_init[1].c_str());
		stop = atoi(elements_init[2].c_str());
		mutation_count = 0;
		for (i = start; i < stop; i++) {
			strcpy(key_init, elements_init[0].c_str());
			strcat(key_init, "-");
			sprintf(temp, "%d", i);
			strcat(key_init, temp);
			sprintf(temp2, "%d", i+1);
			if(variants_map.find(key_init) != variants_map.end()) {
				pfmodel << elements_init[0] << '\t' << temp << '\t' << temp2 << '\t' <<  variants_map[key_init] << endl;
                		mutation_count++;
			}
		}
		pmodel << mutation_count << endl;
		mutation_counts.push_back(mutation_counts);		
	}
	annotation_file.close();
	pmodel.close();
	pfmodel.close();
	system("bwtool extract bed -decimals=12 intersections.bed /fastscratch/jz435/kmerBed/aditya/funseq_wiggle/funseq.bw /dev/stdout | awk '{print $4, $6}' > funseq_intersection_values_raw");
	ifstream funseq_raw("funseq_intersection_values_raw");
	ofstream final_funseq("funseq_intersection_values");
	
	double mutation_count2;
	double funseq_value;
	double normalized_funseq_value;
	while(getline(funseq_raw, line_init)) {
		elements_init.clear();
		elements_init = split(line_init.c_str(), ' ');
		mutation_count2 = atof(elements_init[0].c_str());
		funseq_value = atof(elements_init[1].c_str());
		normalized_funseq_value = mutation_count2 * min(1.0, funseq_value/normalize);
		final_funseq << normalized_funseq_value << endl;
	}
	funseq_raw.close();
	final_funseq.close();
	printf("generating column2\n");
	ifstream counts("intersection_counts");
	ifstream values("funseq_intersection_values");
	ofstream col2("column2");
	int cur_count;
	double cur_value;
	while(getline(counts, line_init)) {
		cur_count = atoi(line_init.c_str());
		//printf("%d\n", cur_count);
		if (cur_count == 0) {
			col2 << "0" << endl;
		} else {
			cur_value = 0.0;
			for (int i = 0; i < cur_count; ++i) {
				getline(values, line_init2);
				cur_value += atof(line_init2.c_str());
			}
			sprintf(temp, "%lf", cur_value);
			col2 << temp << endl;
		}
	}
	counts.close();
	values.close();
	col2.close();
	strcpy(key_init, "awk '{print $4}' ");
        strcat(key_init, avalue);
        strcat(key_init, " > column1");
        system(key_init);
        printf("getting all funseq and prediction values");
	strcpy(key_init, "bwtool extract bed -decimals=12 ");
        strcat(key_init, avalue);
        strcat(key_init, " /fastscratch/jz435/kmerBed/aditya/funseq_wiggle/funseq.bw /dev/stdout | awk '{print $6}' | tr ',' '\n' > funseq_only_values");
        system(key_init);
        if (dflag == 1) {
                strcpy(key_init, "bwtool extract bed -decimals=12 ");
                strcat(key_init, avalue);
                strcat(key_init, " /fastscratch/jz435/kmerBed/aditya/pred_wiggle/predictions.bw /dev/stdout | awk '{print $6}' | tr ',' '\n' > predictions_only_values");
                system(key_init);
        } else {
		non_default_vinas(cvalue, vvalue, avalue);
	}
	ifstream raw("funseq_only_values");
	ofstream normalized("normalized_funseq_only_values");
	while(getline(raw, line_init)) {
		cur_value = min(1, atof(line_init.c_str())/normalize);
		normalized << cur_value << endl;
	}
	raw.close();
	normalized.close();
	system("paste normalized_funseq_only_values predictions_only_values | awk '{print $1*$2}' > multiplied_values");
	system("paste normalized_funseq_only_values predictions_only_values | awk '{print $1*$1*$2}' > statistic_values");
	ifstream counts2(avalue);
	ifstream values2("multiplied_values");
	ofstream col3("column3");
	while(getline(counts2, line_init)) {
		elements_init.clear();
		elements_init = split(line_init, '\t');
                cur_count = atoi(elements_init[2].c_str()) - atoi(elements_init[1].c_str());
                if (cur_count == 0) {
                        col3 << "0" << endl;
                } else {
                        cur_value = 0;
                        for (int i = 0; i < cur_count; ++i) {
                                getline(values2, line_init2);
                                cur_value += atof(line_init2.c_str());
                        }
                        sprintf(temp, "%lf", cur_value);
                        col3 << temp << endl;
                }
        }
	ifstream values3("statistic_values");
	ofstream col4("column4");
	ifstream counts3(avalue);
	while(getline(counts3, line_init)) {
                elements_init.clear();
                elements_init = split(line_init, '\t');
                cur_count = atoi(elements_init[2].c_str()) - atoi(elements_init[1].c_str());
                if (cur_count == 0) {
                        col3 << "0" << endl;
                } else {
                        cur_value = 0;
                        for (int i = 0; i < cur_count; ++i) {
                                getline(values3, line_init2);
                                cur_value += atof(line_init2.c_str());
                        }
                        sprintf(temp, "%lf", cur_value);
                        col4 << temp << endl;
                }
        }

	system("paste column1 column2 column3 column4 > assembly");
	printf("processing assembly\n");
	ifstream assembly("assembly");
	ofstream output(ovalue);
	double cur_col2;
	double cur_col3;
	double cur_col4;
	double delta;
	double result;
	double new_result;
	int line_counter = 0
	while(getline(assembly, line_init)) {
		elements_init.clear();
		elements_init = split(line_init, '\t');
		cur_col2 = atof(elements_init[1].c_str());
		cur_col3 = atof(elements_init[2].c_str());
		cur_col4 = atof(elements_init[3].c_str());
		if (cur_col4 == 0) {
			new_result = 1;
		} 
		if (cur_col2 == 0) {
			new_result = pbinom(mutation_counts[line_counter], )
		}
		else {
			new_result = min(1, exp((-1)*(cur_col3 - cur_col2)*(cur_col3 - cur_col2)/cur_col4));
		}
		if (cur_col3 == 0) {
			result = 1;
			//output << elements_init[0].c_str() << '\t' << "1" << endl;
		} else {
			if (cur_col2 > cur_col3) {
				delta = (cur_col2/cur_col3) - 1;
			}
			else {
				delta = (-1)*(cur_col2/cur_col3) + 1;
			}
			result = pow(exp(1)/(1+delta), delta);
			result = result/(1+delta);
			result = pow(result, cur_col3);
			//output << elements_init[0].c_str() << '\t' << result << endl;
		}
		output << elements_init[0].c_str() << '\t' << result << '\t' << new_result << endl;
		line_counter = line_counter + 1;
	}
	assembly.close();
	output.close();
	return 0;
}
