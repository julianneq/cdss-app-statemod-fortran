from string import Template

# load data
template_name = 'cm2015B_template.rsp'

# Experimental designs
designs = ['CMIPunscaled_SOWs']#['LHsamples_original_1000_AnnQonly','LHsamples_original_200_AnnQonly',\
			#'LHsamples_narrowed_1000_AnnQonly','LHsamples_narrowed_200_AnnQonly',\
            #'LHsamples_wider_1000_AnnQonly','LHsamples_wider_200_AnnQonly','CMIP_SOWs','Paleo_SOWs']
nSamples = [97]#[1000, 200, 1000, 200, 1000, 200, 209, 366]

# create RSP files
with open(template_name, 'r') as T:
    template = Template(T.read())
    for k, design in enumerate(designs):
        for i in range(nSamples[k]+1):
            for j in range(10):
                d = {}
                d['IWR'] = 'cm2015B_S' + str(i) + '_' + str(j+1) + '.iwr'
                d['XBM'] = 'cm2015x_S' + str(i) + '_' + str(j+1) + '.xbm'
                d['DDM'] = 'cm2015B_S' + str(i) + '_' + str(j+1) + '.ddm'
                S1 = template.safe_substitute(d)
                with open('./../../' + design + '/cm2015B_S' + str(i) + '_' + str(j+1) + '.rsp', 'w') as f1:
                    f1.write(S1)
                    
                f1.close()