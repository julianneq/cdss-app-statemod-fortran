NSAMPLES=(100 1000)
METHOD=Latin
JAVA_ARGS="-cp MOEAFramework-2.4-Demo.jar"

# Generate the parameter samples
for NSAMPLE in ${NSAMPLES}
do
	java ${JAVA_ARGS} \
	    org.moeaframework.analysis.sensitivity.SampleGenerator \
	    --method ${METHOD} --n ${NSAMPLE} --p uncertain_params_original.txt \
	    --o LHsamples_original_${NSAMPLE}.txt
	java ${JAVA_ARGS} \
	    org.moeaframework.analysis.sensitivity.SampleGenerator \
	    --method ${METHOD} --n ${NSAMPLE} --p uncertain_params_wider.txt \
	    --o LHsamples_wider_${NSAMPLE}.txt
done

