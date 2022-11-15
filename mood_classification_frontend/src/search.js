import React from 'react'
import TextField from '@material-ui/core/TextField';
import Autocomplete, { createFilterOptions } from '@material-ui/lab/Autocomplete';

const Searchbar = () => {

    const filterOptions = createFilterOptions({
        matchFrom: 'start',
        stringify: option => option,
    });
    const printData = (val) =>
    {
        console.log('aawa', val)
    }
// Sample options for search box
    const myOptions = ['One Number', 'Two Number', 'Five Number',
        'This is a sample text', 'Dummy text', 'Dropdown option teet',
        'Hello text', 'Welcome to text field'];

    return (
        <div style={{ marginLeft: '40%', marginTop: '60px' }}>
            <h3>Greetings from GeeksforGeeks!</h3>
            <Autocomplete
                style={{ width: 500 }}
                freeSolo
                filterOptions={filterOptions}
                options={myOptions}
                renderInput={(params) => (
                    <TextField {...params}
                               variant="outlined"
                               label="Search Box"
                               onChange={() => printData}
                    />
                )}
            />
        </div>
    );
}

export default Searchbar
